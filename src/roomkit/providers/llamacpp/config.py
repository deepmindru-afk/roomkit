"""llama.cpp provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from roomkit.providers.llamacpp._builds import ASSETS


class LlamaCppConfig(BaseModel):
    """A local model that RoomKit runs itself, through llama.cpp's ``llama-server``.

    Only ``model`` is required. On first use the provider downloads the
    llama.cpp build for this machine (checked against a pinned SHA-256) and the
    model, starts ``llama-server`` on a free local port, and stops it when the
    provider is closed.

    Attributes:
        model: The GGUF model to run: a Hugging Face ``repo:quant`` reference
            (``"unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"``), which
            ``llama-server`` downloads and caches itself, or the path of a
            local ``.gguf`` file.
        context_size: Context window in tokens (``-c``). Tool definitions and
            tool results count against it.
        gpu_layers: Layers offloaded to the GPU (``-ngl``). ``None`` lets
            llama.cpp fit as many as the GPU holds; ``0`` runs on the CPU only.
        max_tokens: Maximum tokens in one response.
        temperature: Sampling temperature.
        enable_thinking: Turn a reasoning model's thinking on or off through
            its chat template. ``None`` keeps the model's default.
        binary: A ``llama-server`` to use instead of the pinned download
            (your own build, a distribution package): a path, or a name looked
            up on the ``PATH``. A ``llama-server`` merely present on the
            machine is never picked up on its own.
        variant: Force a download variant (``"linux-x64-cpu"``,
            ``"linux-x64-cuda-13"``, ``"macos-arm64"``…) instead of the one
            detected for this machine.
        cache_dir: Where downloaded llama.cpp builds are kept. Defaults to
            ``~/.cache/roomkit/llama.cpp``.
        port: Local port for ``llama-server``. ``None`` picks a free one.
        startup_timeout: Seconds to wait for the server to be ready. The first
            start downloads the model, so the default is generous.
        timeout: HTTP timeout of one request, in seconds.
        extra_args: More ``llama-server`` arguments, passed as given
            (``["--threads", "8"]``).
    """

    model: str
    context_size: int = Field(default=8192, gt=0)
    gpu_layers: int | None = Field(default=None, ge=0)
    max_tokens: int = Field(default=1024, gt=0)
    temperature: float = 0.7
    enable_thinking: bool | None = None
    binary: str | None = None
    variant: str | None = None
    cache_dir: str | None = None
    port: int | None = Field(default=None, gt=0, lt=65536)
    startup_timeout: float = Field(default=1800.0, gt=0)
    timeout: float = Field(default=120.0, gt=0)
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("model")
    @classmethod
    def _model_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must name a GGUF: 'repo:quant' or a .gguf path")
        return value

    @field_validator("variant")
    @classmethod
    def _known_variant(cls, value: str | None) -> str | None:
        if value is not None and value not in ASSETS:
            raise ValueError(f"unknown variant {value!r}; one of {sorted(ASSETS)}")
        return value
