"""Pin the llama.cpp build the llamacpp provider downloads.

Reads a llama.cpp GitHub release and rewrites
``src/roomkit/providers/llamacpp/_builds.py`` with the asset and SHA-256 of
every variant the provider knows how to pick. By default it pins the latest
stable release (``vX.Y.Z``): a stable release carries no binaries, only a
``nightly-tag.txt`` naming the ``bNNNN`` build it is, whose binaries are pinned.

    uv run python scripts/update_llamacpp_build.py            # latest stable
    uv run python scripts/update_llamacpp_build.py v0.8.0     # a given stable
    uv run python scripts/update_llamacpp_build.py b11960     # a given build
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
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
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310 - fixed https host
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{url}: {exc.code} {exc.reason}") from None


def _text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:  # nosec B310 - a GitHub release asset
        return response.read().decode().strip()


def _release(tag: str | None) -> tuple[dict, str | None]:
    """The ``bNNNN`` release to pin, and the stable release it is, if any.

    No *tag* is the latest stable release (``releases/latest``: llama.cpp
    marks every ``bNNNN`` build a prerelease).
    """
    if tag and re.fullmatch(r"b\d+", tag):
        return _get(f"{API}/tags/{tag}"), None  # type: ignore[return-value]
    stable: dict = _get(f"{API}/tags/{tag}" if tag else f"{API}/latest")  # type: ignore[assignment]
    name = stable["tag_name"]
    urls = [a["browser_download_url"] for a in stable["assets"] if a["name"] == "nightly-tag.txt"]
    if len(urls) != 1:
        raise SystemExit(f"{name}: no nightly-tag.txt naming its build")
    build = _text(urls[0])
    if not re.fullmatch(r"b\d+", build):
        raise SystemExit(f"{name}: nightly-tag.txt names {build!r}, not a bNNNN build")
    return _get(f"{API}/tags/{build}"), name  # type: ignore[return-value]


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
    release, stable = _release(sys.argv[1] if len(sys.argv) > 1 else None)
    tag = release["tag_name"]
    missing = [
        a["name"] for a in release["assets"] if not (a.get("digest") or "").startswith("sha256:")
    ]
    if missing:
        raise SystemExit(f"{tag}: no SHA-256 digest published for {missing}")
    assets = {a["name"]: a["digest"].removeprefix("sha256:") for a in release["assets"]}
    rows = []
    for variant, (pattern, cudart) in VARIANTS.items():
        name, digest = _pick(assets, f"llama-{tag}-bin-", pattern)
        extra = _pick_cudart(assets, tag, pattern) if cudart else None
        rows.append((variant, name, digest, extra))

    pinned = f"llama.cpp {stable} (build {tag})" if stable else f"llama.cpp build {tag}"
    lines = [
        '"""The llama.cpp build the llamacpp provider downloads — generated, do not edit.',
        "",
        f"{pinned}.",
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
    print(f"pinned {pinned}: {len(rows)} variants → {TARGET}")


if __name__ == "__main__":
    main()
