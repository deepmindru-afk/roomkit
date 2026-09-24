"""llama.cpp provider — a local GGUF model RoomKit runs itself.

``pip install roomkit[llamacpp]``, then::

    from roomkit.providers.llamacpp import LlamaCppAIProvider, LlamaCppConfig

    ai = LlamaCppAIProvider(LlamaCppConfig(model="unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"))

No server to install or start: the provider downloads the pinned llama.cpp
build for this machine and the model, runs ``llama-server`` on a local port,
and stops it on :meth:`~LlamaCppAIProvider.close`.
"""

from __future__ import annotations

from roomkit.providers.llamacpp.ai import LlamaCppAIProvider
from roomkit.providers.llamacpp.config import LlamaCppConfig

__all__ = ["LlamaCppAIProvider", "LlamaCppConfig"]
