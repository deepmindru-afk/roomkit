"""Local models for examples: check they are downloaded before anything starts."""

from __future__ import annotations

import logging
import sys

from roomkit.local_models import default_models_dir, is_downloaded

logger = logging.getLogger(__name__)


def require_local_models(*names: str) -> None:
    """Exit with the command that downloads whichever of *names* are missing.

    The models live in ``$ROOMKIT_MODELS_DIR``, else ``~/.cache/roomkit/models``.
    """
    missing = [name for name in names if not is_downloaded(name)]
    if missing:
        logger.error(
            "Models not downloaded in %s. Download them once, while online:\n"
            "    python -m roomkit.local_models download %s",
            default_models_dir(),
            " ".join(missing),
        )
        sys.exit(1)
