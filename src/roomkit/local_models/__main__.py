"""``python -m roomkit.local_models``: list, download and locate local models."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from roomkit.local_models import (
    KINDS,
    LOCAL_MODELS,
    LocalModel,
    ModelDownloadError,
    ModelNotDownloadedError,
    default_models_dir,
    download_local_model,
    find_local_model,
    is_downloaded,
    local_model_dir,
)

logger = logging.getLogger("roomkit.local_models")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        return _list(args.kind, args.models_dir)
    try:
        models = [find_local_model(name) for name in args.names]
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    if args.command == "download":
        return _download(models, args.models_dir)
    return _path(models, args.models_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m roomkit.local_models",
        description="Download local models once, while online; RoomKit then runs them offline.",
    )
    # On every subcommand, after its name: `download --models-dir DIR NAME`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--models-dir",
        help=f"Where models live (default: $ROOMKIT_MODELS_DIR, else {default_models_dir()})",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    list_p = sub.add_parser(
        "list", parents=[common], help="The known models, and which are downloaded"
    )
    list_p.add_argument("--kind", choices=KINDS, help="Only the models for this provider config")
    download_p = sub.add_parser(
        "download", parents=[common], help="Download models (skips the ones already there)"
    )
    download_p.add_argument("names", nargs="+", metavar="NAME")
    path_p = sub.add_parser("path", parents=[common], help="Print a downloaded model's directory")
    path_p.add_argument("names", nargs="+", metavar="NAME")
    return parser


def _list(kind: str | None, models_dir: str | None) -> int:
    root = default_models_dir() if models_dir is None else Path(models_dir)
    rows = [model for model in LOCAL_MODELS if kind is None or model.kind == kind]
    width = max(len(model.name) for model in rows)
    for model in rows:
        mark = "*" if is_downloaded(model, root) else " "
        logger.info(
            "%s %-*s  %-20s %5d MB  %s",
            mark,
            width,
            model.name,
            model.kind,
            model.size_mb,
            model.description,
        )
    logger.info("\n* downloaded in %s", root)
    return 0


def _download(models: Sequence[LocalModel], models_dir: str | None) -> int:
    for model in models:
        try:
            path = download_local_model(model, models_dir)
        except ModelDownloadError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("%s", path)
    return 0


def _path(models: Sequence[LocalModel], models_dir: str | None) -> int:
    for model in models:
        try:
            logger.info("%s", local_model_dir(model, models_dir))
        except ModelNotDownloadedError as exc:
            logger.error("%s", exc)
            return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    sys.exit(main())
