"""Fetch, verify and install a local model's files."""

from __future__ import annotations

import hashlib
import logging
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.local_models._model import LocalModel, ModelFile

logger = logging.getLogger("roomkit.local_models")

# Written last into a model's directory: it lists the SHA-256 of every file the
# directory was built from, so a model whose pinned files changed reads as not
# downloaded instead of silently running the previous version.
_COMPLETE_MARKER = ".roomkit-complete"
_USER_AGENT = "roomkit-local-models"
_CHUNK_SIZE = 1 << 20


class ModelDownloadError(RuntimeError):
    """A model file could not be fetched, or did not match its pinned SHA-256."""


def is_installed(target: Path, model: LocalModel) -> bool:
    """Whether ``target`` holds a complete copy of exactly this model's files."""
    marker = target / _COMPLETE_MARKER
    return marker.is_file() and marker.read_text().split() == _digests(model.files)


def install(target: Path, model: LocalModel) -> None:
    """Download ``model`` into ``target``; ``target`` only ever appears complete.

    Every file lands in a staging directory beside ``target`` and is checked
    against its pinned SHA-256 before the staging directory is renamed into
    place. An older copy at ``target`` (other pinned files) is replaced.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{model.name}-") as tmp:
        staging = Path(tmp) / "model"
        staging.mkdir()
        for file in model.files:
            if file.is_archive:
                archive = Path(tmp) / file.name
                _fetch(file, archive)
                _extract(archive, staging)
                archive.unlink()
            else:
                _fetch(file, staging / file.name)
        (staging / _COMPLETE_MARKER).write_text("\n".join(_digests(model.files)) + "\n")
        if target.exists():
            # An older version: move it into the temporary directory, which
            # removes it on exit, so the rename below has a free name.
            target.rename(Path(tmp) / "previous")
        try:
            staging.rename(target)
        except OSError as exc:
            # Another process installed the same model meanwhile: use it.
            if not is_installed(target, model):
                message = f"cannot install {model.name} into {target}: {exc}"
                raise ModelDownloadError(message) from exc
            logger.info("%s was installed concurrently", model.name)


def _digests(files: Sequence[ModelFile]) -> list[str]:
    return [file.sha256 for file in files]


def _fetch(file: ModelFile, dest: Path) -> None:
    logger.info("Downloading %s", file.url)
    digest = hashlib.sha256()
    request = urllib.request.Request(file.url, headers={"User-Agent": _USER_AGENT})
    try:
        # ModelFile refuses any URL that is not https://.
        with (
            urllib.request.urlopen(request, timeout=60) as response,  # nosec B310
            dest.open("wb") as out,
        ):
            while chunk := response.read(_CHUNK_SIZE):
                digest.update(chunk)
                out.write(chunk)
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise ModelDownloadError(f"cannot download {file.url}: {exc}") from exc
    if digest.hexdigest() != file.sha256:
        dest.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"{file.name}: SHA-256 {digest.hexdigest()} does not match the pinned "
            f"{file.sha256}; refusing to install it"
        )


def _extract(archive: Path, dest: Path) -> None:
    try:
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    resolved = (dest / member).resolve()
                    if not resolved.is_relative_to(dest.resolve()):
                        raise ModelDownloadError(f"{archive.name}: member {member!r} escapes it")
                zf.extractall(dest)  # nosec B202  # every member path checked above
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(dest, filter="data")  # the "data" filter refuses escapes
    except (tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ModelDownloadError(f"cannot extract {archive.name}: {exc}") from exc
