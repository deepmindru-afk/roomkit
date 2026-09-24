"""Pin the llama.cpp build the llamacpp provider downloads.

Reads a llama.cpp GitHub release (the newest ``bNNNN`` build by default, or the
tag given) and rewrites ``src/roomkit/providers/llamacpp/_builds.py`` with the
asset and SHA-256 of every variant the provider knows how to pick.

    uv run python scripts/update_llamacpp_build.py            # newest build
    uv run python scripts/update_llamacpp_build.py b11160     # a given build
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
TARGET = Path(__file__).resolve().parent.parent / "src/roomkit/providers/llamacpp/_builds.py"

# variant → (regex over the asset name after "llama-<tag>-bin-", ships a CUDA runtime).
# CUDA assets carry the toolkit's minor version (12.8, 13.4…), which moves between
# builds: the regex keeps the major, the table records the exact file. A CUDA
# runtime archive has the same suffix as its build.
VARIANTS: dict[str, tuple[str, bool]] = {
    "linux-x64-cpu": (r"ubuntu-x64\.tar\.gz", False),
    "linux-x64-cuda-12": (r"ubuntu-cuda-12\.\d+-x64\.tar\.gz", True),
    "linux-x64-cuda-13": (r"ubuntu-cuda-13\.\d+-x64\.tar\.gz", True),
    "linux-x64-vulkan": (r"ubuntu-vulkan-x64\.tar\.gz", False),
    "linux-arm64-cpu": (r"ubuntu-arm64\.tar\.gz", False),
    "linux-arm64-cuda-13": (r"ubuntu-cuda-13\.\d+-arm64\.tar\.gz", True),
    "macos-arm64": (r"macos-arm64\.tar\.gz", False),
    "macos-x64": (r"macos-x64\.tar\.gz", False),
    "windows-x64-cpu": (r"win-cpu-x64\.zip", False),
    "windows-x64-cuda-12": (r"win-cuda-12\.\d+-x64\.zip", True),
    "windows-x64-cuda-13": (r"win-cuda-13\.\d+-x64\.zip", True),
    "windows-arm64-cpu": (r"win-cpu-arm64\.zip", False),
}


def _get(url: str) -> object:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310 - fixed https host
        return json.load(response)


def _release(tag: str | None) -> dict:
    if tag:
        return _get(f"{API}/tags/{tag}")  # type: ignore[return-value]
    for release in _get(f"{API}?per_page=20"):  # type: ignore[union-attr]
        # llama.cpp publishes every bNNNN build as a prerelease: the newest one
        # is the build to pin, whatever the flag says.
        if re.fullmatch(r"b\d+", release["tag_name"]) and not release["draft"]:
            return release
    raise SystemExit("no bNNNN release among the 20 newest")


def _pick(assets: dict[str, str], prefix: str, pattern: str) -> tuple[str, str]:
    names = [n for n in assets if re.fullmatch(re.escape(prefix) + pattern, n)]
    if len(names) != 1:
        raise SystemExit(f"expected one asset for {prefix}{pattern}, found {names}")
    return names[0], assets[names[0]]


def _pick_cudart(assets: dict[str, str], tag: str, pattern: str) -> tuple[str, str]:
    # The Linux runtimes carry the build tag, the Windows ones do not.
    for prefix in (f"cudart-llama-{tag}-bin-", "cudart-llama-bin-"):
        if any(re.fullmatch(re.escape(prefix) + pattern, n) for n in assets):
            return _pick(assets, prefix, pattern)
    raise SystemExit(f"no CUDA runtime for {pattern}")


def main() -> None:
    release = _release(sys.argv[1] if len(sys.argv) > 1 else None)
    tag = release["tag_name"]
    assets = {a["name"]: a["digest"].removeprefix("sha256:") for a in release["assets"]}
    rows = []
    for variant, (pattern, cudart) in VARIANTS.items():
        name, digest = _pick(assets, f"llama-{tag}-bin-", pattern)
        extra = _pick_cudart(assets, tag, pattern) if cudart else None
        rows.append((variant, name, digest, extra))

    lines = [
        '"""The llama.cpp build the llamacpp provider downloads — generated, do not edit.',
        "",
        "Regenerate with ``uv run python scripts/update_llamacpp_build.py``.",
        '"""',
        "",
        "# ruff: noqa: E501",
        "",
        "from __future__ import annotations",
        "",
        f'BUILD = "{tag}"',
        "",
        "# variant -> ((archive, sha256), ...): the build, then its CUDA runtime if any.",
        "ASSETS: dict[str, tuple[tuple[str, str], ...]] = {",
    ]
    for variant, name, digest, extra in rows:
        lines.append(f'    "{variant}": (')
        lines.append(f'        ("{name}", "{digest}"),')
        if extra:
            lines.append(f'        ("{extra[0]}", "{extra[1]}"),')
        lines.append("    ),")
    lines.append("}")
    TARGET.write_text("\n".join(lines) + "\n")
    print(f"pinned llama.cpp {tag}: {len(rows)} variants → {TARGET}")


if __name__ == "__main__":
    main()
