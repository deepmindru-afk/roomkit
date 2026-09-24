"""Find or download the ``llama-server`` executable for this machine.

By default the pinned llama.cpp build (:mod:`._builds`), downloaded from
GitHub into the cache and verified against its SHA-256 before anything in it is
extracted or run. A ``llama-server`` already on the machine is used only when
``binary`` names it: one found on the ``PATH`` by chance may be old, built
without the GPU, or a broken wrapper, and would decide the behaviour silently.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
import shutil
import subprocess  # nosec B404 - fixed argument list, no shell
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from roomkit.providers.ai.base import ProviderError
from roomkit.providers.llamacpp._builds import ASSETS, BUILD

logger = logging.getLogger("roomkit.providers.llamacpp")

_RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{build}/{name}"
_SERVER_NAMES = ("llama-server", "llama-server.exe")
_COMPLETE_MARKER = ".complete"


@dataclass(frozen=True)
class LlamaBinary:
    """A runnable ``llama-server`` and the directories its libraries live in."""

    executable: Path
    library_dirs: tuple[Path, ...] = ()


def resolve_binary(
    *, binary: str | None, variant: str | None, cache_dir: str | None
) -> LlamaBinary:
    """The ``llama-server`` to run, downloading the pinned build if needed.

    Blocking (it may download): call it from a thread.
    """
    if binary is not None:
        path = Path(binary).expanduser()
        found = path if path.is_file() else _which(binary)
        if found is None:
            raise _error(f"llama-server not found: binary={binary!r} is no file nor on the PATH")
        logger.info("Using llama-server %s", found)
        return LlamaBinary(found)
    chosen = variant or detect_variant()
    root = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
    logger.info("llama.cpp %s, variant %s%s", BUILD, chosen, "" if variant else " (detected)")
    return _cached_build(root / BUILD / chosen, chosen)


def detect_variant() -> str:
    """The pinned build that runs on this OS, CPU and GPU driver."""
    system = platform.system()
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if arch is None:
        raise _error(f"no llama.cpp build for CPU {machine!r}; set binary= to your own")
    if system == "Darwin":
        return f"macos-{arch}"  # the arm64 build runs on Metal
    os_name = {"Linux": "linux", "Windows": "windows"}.get(system)
    if os_name is None:
        raise _error(f"no llama.cpp build for {system!r}; set binary= to your own")
    cuda = cuda_driver_major()
    for major in (13, 12):
        candidate = f"{os_name}-{arch}-cuda-{major}"
        if cuda is not None and cuda >= major and candidate in ASSETS:
            return candidate
    return f"{os_name}-{arch}-cpu"


def cuda_driver_major() -> int | None:
    """The highest CUDA major version the installed NVIDIA driver supports, if any."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run(  # nosec B603 - fixed argument list
            [smi], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.info("nvidia-smi failed (%s): using a build without CUDA", exc)
        return None
    match = re.search(r"CUDA Version:\s*(\d+)", out)
    if match is None:
        logger.info("nvidia-smi reports no CUDA version: using a build without CUDA")
        return None
    return int(match.group(1))


def _which(name: str) -> Path | None:
    found = shutil.which(name)
    return Path(found) if found else None


def _default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "roomkit" / "llama.cpp"


def _cached_build(target: Path, variant: str) -> LlamaBinary:
    if not (target / _COMPLETE_MARKER).is_file():
        _download_build(target, variant)
    return _locate(target)


def _download_build(target: Path, variant: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{variant}-") as tmp:
        staging = Path(tmp) / "build"
        staging.mkdir()
        for name, sha256 in ASSETS[variant]:
            archive = Path(tmp) / name
            _download(_RELEASE_URL.format(build=BUILD, name=name), archive, sha256)
            _extract(archive, staging)
        (staging / _COMPLETE_MARKER).write_text(f"{BUILD} {variant}\n")
        try:
            staging.rename(target)  # atomic: target only ever appears complete
        except OSError as exc:
            # Another process installed the same build meanwhile: use it.
            if not (target / _COMPLETE_MARKER).is_file():
                raise _error(f"cannot install llama.cpp into {target}: {exc}") from exc
            logger.info("llama.cpp %s (%s) was installed concurrently", BUILD, variant)
            return
    logger.info("llama.cpp %s (%s) installed in %s", BUILD, variant, target)


def _download(url: str, dest: Path, sha256: str) -> None:
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required to download llama.cpp. "
            "Install it with: pip install roomkit[llamacpp]"
        ) from exc
    logger.info("Downloading %s", url)
    digest = hashlib.sha256()
    with (
        httpx.stream("GET", url, follow_redirects=True, timeout=60) as response,
        dest.open("wb") as out,
    ):
        response.raise_for_status()
        for chunk in response.iter_bytes(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != sha256:
        dest.unlink(missing_ok=True)
        raise _error(
            f"{dest.name}: SHA-256 {digest.hexdigest()} does not match the pinned {sha256}; "
            "refusing to install it"
        )


def _extract(archive: Path, dest: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                resolved = (dest / member).resolve()
                if not resolved.is_relative_to(dest.resolve()):
                    raise _error(f"{archive.name}: member {member!r} escapes the archive")
            zf.extractall(dest)  # nosec B202 - every member path checked above
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")  # nosec B202 - "data" filter refuses escapes


def _locate(root: Path) -> LlamaBinary:
    for name in _SERVER_NAMES:
        found = sorted(root.rglob(name))
        if found:
            libraries = {p.parent for p in root.rglob("*") if _is_library(p.name)}
            return LlamaBinary(found[0], tuple(sorted(libraries)))
    raise _error(f"no llama-server in {root}; delete the directory to download it again")


def _is_library(name: str) -> bool:
    return name.endswith((".dll", ".dylib", ".so")) or ".so." in name


def _error(message: str) -> ProviderError:
    return ProviderError(message, retryable=False, provider="llamacpp")
