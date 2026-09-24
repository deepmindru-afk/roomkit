"""The ``llama-server`` process a :class:`LlamaCppAIProvider` owns."""

from __future__ import annotations

import asyncio
import atexit
import collections
import logging
import os
import socket
import sys
import time
import weakref
from pathlib import Path

from roomkit.providers.ai.base import ProviderError
from roomkit.providers.llamacpp.binary import LlamaBinary, resolve_binary
from roomkit.providers.llamacpp.config import LlamaCppConfig

logger = logging.getLogger("roomkit.providers.llamacpp")

_HOST = "127.0.0.1"
_LOG_TAIL_LINES = 40
_HEALTH_POLL_S = 0.25
_STOP_GRACE_S = 10.0

# Servers still running when the interpreter exits, killed by the atexit hook
# below so a script that never closes its provider leaves no llama-server behind.
_live: weakref.WeakSet[LlamaServer] = weakref.WeakSet()


class LlamaServer:
    """Start ``llama-server`` once, wait until it serves, stop it on demand."""

    def __init__(self, config: LlamaCppConfig) -> None:
        self._config = config
        self._port = config.port or _free_port()
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._tail: collections.deque[str] = collections.deque(maxlen=_LOG_TAIL_LINES)
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        """The OpenAI-compatible endpoint the server answers on."""
        return f"http://{_HOST}:{self._port}/v1"

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        """Start the server and return once it answers ``/health``. Idempotent."""
        async with self._lock:
            if self.running:
                return
            config = self._config
            binary = await asyncio.to_thread(
                resolve_binary,
                binary=config.binary,
                variant=config.variant,
                cache_dir=config.cache_dir,
            )
            args = self._arguments(binary)
            if self._config.port is not None:
                _check_port_free(self._port)
            logger.info("Starting llama-server for %s on port %d", config.model, self._port)
            self._tail.clear()
            self._process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=_environment(binary),
            )
            _live.add(self)
            self._reader = asyncio.create_task(self._read_output(self._process))
            try:
                await self._wait_ready(self._process)
            except BaseException:
                await self._stop_process()
                raise
            logger.info("llama-server ready at %s", self.base_url)

    async def stop(self) -> None:
        """Stop the server if it runs: terminate, then kill after a grace period."""
        async with self._lock:
            await self._stop_process()

    def _arguments(self, binary: LlamaBinary) -> list[str]:
        config = self._config
        model = config.model
        is_file = model.endswith(".gguf")
        if is_file and not Path(model).expanduser().is_file():
            raise ProviderError(f"model file not found: {model}", provider="llamacpp")
        args = [
            str(binary.executable),
            *(["-m", str(Path(model).expanduser())] if is_file else ["-hf", model]),
            "--host",
            _HOST,
            "--port",
            str(self._port),
            "--jinja",  # the model's own chat template: native tool calls
            "-c",
            str(config.context_size),
        ]
        if config.gpu_layers is not None:
            args += ["-ngl", str(config.gpu_layers)]
        return args + list(config.extra_args)

    async def _wait_ready(self, process: asyncio.subprocess.Process) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for LlamaCppAIProvider. "
                "Install it with: pip install roomkit[llamacpp]"
            ) from exc

        deadline = time.monotonic() + self._config.startup_timeout
        async with httpx.AsyncClient(timeout=5) as client:
            while time.monotonic() < deadline:
                if process.returncode is not None:
                    await self._drain_output()
                    raise self._failure(f"llama-server exited with code {process.returncode}")
                try:
                    response = await client.get(f"http://{_HOST}:{self._port}/health")
                    if response.status_code == 200:
                        return
                except httpx.TransportError:
                    pass  # not listening yet (downloading or loading the model)
                await asyncio.sleep(_HEALTH_POLL_S)
        raise self._failure(
            f"llama-server not ready after {self._config.startup_timeout:.0f}s "
            "(raise startup_timeout if the model is still downloading)"
        )

    async def _read_output(self, process: asyncio.subprocess.Process) -> None:
        if process.stdout is None:
            return
        async for raw in process.stdout:
            line = raw.decode(errors="replace").rstrip()
            if line:
                self._tail.append(line)
                logger.debug("llama-server: %s", line)

    async def _drain_output(self) -> None:
        """Let the reader take the last lines of a process that just exited."""
        if self._reader is not None:
            await asyncio.wait({self._reader}, timeout=2)

    async def _stop_process(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), _STOP_GRACE_S)
            except TimeoutError:
                process.kill()
                await process.wait()
            logger.info("llama-server stopped")
        if self._reader is not None:
            await asyncio.gather(self._reader, return_exceptions=True)
            self._reader = None
        _live.discard(self)

    def _failure(self, reason: str) -> ProviderError:
        tail = "\n".join(self._tail) or "(no output)"
        return ProviderError(f"{reason}. Last output:\n{tail}", provider="llamacpp")

    def _kill_now(self) -> None:
        if self._process is not None and self._process.returncode is None:
            self._process.kill()


def _check_port_free(port: int) -> None:
    """Refuse a configured port already taken: its ``/health`` would pass for ours."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((_HOST, port))
        except OSError as exc:
            raise ProviderError(
                f"port {port} is already in use on {_HOST}; pick another or leave port unset",
                provider="llamacpp",
            ) from exc


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_HOST, 0))
        return int(sock.getsockname()[1])


def _environment(binary: LlamaBinary) -> dict[str, str]:
    """The environment with the build's shared libraries on the loader path."""
    env = dict(os.environ)
    if not binary.library_dirs:
        return env
    dirs = os.pathsep.join(str(d) for d in binary.library_dirs)
    var = {"win32": "PATH", "darwin": "DYLD_LIBRARY_PATH"}.get(sys.platform, "LD_LIBRARY_PATH")
    env[var] = dirs + (os.pathsep + env[var] if env.get(var) else "")
    return env


@atexit.register
def _kill_live_servers() -> None:
    for server in list(_live):
        server._kill_now()
