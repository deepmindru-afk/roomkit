"""Audio mixer providers for PCM frame mixing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.voice.pipeline.mixer.base import MixerProvider
from roomkit.voice.pipeline.mixer.python import PythonMixerProvider

if TYPE_CHECKING:
    from roomkit.voice.pipeline.mixer.numpy import NumpyMixerProvider
else:
    # NumpyMixerProvider loads on first use: numpy is optional.
    __getattr__ = lazy_exports(
        __name__, {"NumpyMixerProvider": "roomkit.voice.pipeline.mixer.numpy"}
    )

__all__ = [
    "MixerProvider",
    "NumpyMixerProvider",
    "PythonMixerProvider",
]
