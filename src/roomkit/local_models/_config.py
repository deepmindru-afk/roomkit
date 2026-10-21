"""Building a provider config from a downloaded local model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Self, cast

from roomkit.local_models._registry import local_model_config


class FromLocalModel:
    """Mixin for a provider config that a local model can fill in.

    The config class names the kind of model it takes in ``local_model_kind``.
    """

    local_model_kind: ClassVar[str]

    @classmethod
    def from_local_model(
        cls, name: str, *, models_dir: str | Path | None = None, **overrides: Any
    ) -> Self:
        """The config for a downloaded local model, by the model's name.

        Reads no network. Download the model once beforehand with
        ``python -m roomkit.local_models download <name>``.

        Args:
            name: The model (``python -m roomkit.local_models list`` names them).
            models_dir: Where models live; default ``$ROOMKIT_MODELS_DIR``,
                else ``~/.cache/roomkit/models``.
            **overrides: Any other field of the config (``provider="cuda"``,
                ``num_threads=4``); it wins over the model's own settings.

        Raises:
            ModelNotDownloadedError: The model is not downloaded; the message
                names the command that downloads it.
            ValueError: No model has that name, or it is for another config.
        """
        values = local_model_config(name, kind=cls.local_model_kind, models_dir=models_dir)
        # A dataclass's generated __init__ takes the fields; the mixin cannot say so.
        return cast(Callable[..., Self], cls)(**{**values, **overrides})
