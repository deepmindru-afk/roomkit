"""The llama.cpp build the llamacpp provider downloads — generated, do not edit.

llama.cpp v0.8.0 (build b11960).

Regenerate with ``uv run python scripts/update_llamacpp_build.py``.
"""

# ruff: noqa: E501

from __future__ import annotations

BUILD = "b11960"

# variant -> ((archive, sha256), ...): the build, then its CUDA runtime if any.
ASSETS: dict[str, tuple[tuple[str, str], ...]] = {
    "linux-x64-cpu": (
        (
            "llama-b11960-bin-ubuntu-x64.tar.gz",
            "31cc3a6bc638ac6c48dc4b1b74888f1eaadf3d14227c1e5f67daf9a49541d97a",
        ),
    ),
    "linux-x64-cuda-12": (
        (
            "llama-b11960-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "64d57beb9abb637c177c6afb313760a0c0782224cac71e74869eed5b8f9a1ea4",
        ),
        (
            "cudart-llama-b11960-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "7c4d5974bd39649df649649e5d71950688a70e37aaae00bf11afbb656ea2cc56",
        ),
    ),
    "linux-x64-cuda-13": (
        (
            "llama-b11960-bin-ubuntu-cuda-13.4-x64.tar.gz",
            "0cf758b28483593a5a15e0d2c9b686c40840d1f8be9b5bb2417f04a053113907",
        ),
        (
            "cudart-llama-b11960-bin-ubuntu-cuda-13.4-x64.tar.gz",
            "756e14988abb5daa60d25f27a9766a9a78f04c88647c4b92a88fb749488d72f7",
        ),
    ),
    "linux-x64-vulkan": (
        (
            "llama-b11960-bin-ubuntu-vulkan-x64.tar.gz",
            "164199e978304bb32a8a6c24b8efa015528ddf286821596eab676c90b375feac",
        ),
    ),
    "linux-arm64-cpu": (
        (
            "llama-b11960-bin-ubuntu-arm64.tar.gz",
            "0065ecfbffe8a790bb0d28059a3320cdc8f723b7f3f248cdc8bfc04790927715",
        ),
    ),
    "linux-arm64-cuda-13": (
        (
            "llama-b11960-bin-ubuntu-cuda-13.4-arm64.tar.gz",
            "a66743e250e2975d7181b9f51048bfbea7903106035b711bea5c552360323452",
        ),
        (
            "cudart-llama-b11960-bin-ubuntu-cuda-13.4-arm64.tar.gz",
            "4b20f8c8dab2eb49c9f46ab93e11858877f1c031696f665ed5584b045187d0a0",
        ),
    ),
    "macos-arm64": (
        (
            "llama-b11960-bin-macos-arm64.tar.gz",
            "58e5a5e1d7c2bf66cedd9c29e0323f3e570660a70a1e2156d32eed08f4cc1e3c",
        ),
    ),
    "macos-x64": (
        (
            "llama-b11960-bin-macos-x64.tar.gz",
            "d1e6c9ddcaed9a3b1bb8f3d946bdb3481c7b919357a4029dcb747762397f1da5",
        ),
    ),
    "windows-x64-cpu": (
        (
            "llama-b11960-bin-win-cpu-x64.zip",
            "73f962f82850d244db47eaacb9d913637027b82399124324298576d58c5f4287",
        ),
    ),
    "windows-x64-cuda-12": (
        (
            "llama-b11960-bin-win-cuda-12.4-x64.zip",
            "8d9d819261de10cc2f03921b8b79c779c5c458e21edb60c170ee30a7d462d684",
        ),
        (
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
            "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
        ),
    ),
    "windows-x64-cuda-13": (
        (
            "llama-b11960-bin-win-cuda-13.4-x64.zip",
            "31c75ea62b80f0816c855ea783ef68757bc2d48028edaa57d904e4982bddea4a",
        ),
        (
            "cudart-llama-bin-win-cuda-13.4-x64.zip",
            "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
        ),
    ),
    "windows-arm64-cpu": (
        (
            "llama-b11960-bin-win-cpu-arm64.zip",
            "77349357c38da235bb0eed37bcc5ecf3efef21bb5dc42c2269589c0e71cfa00f",
        ),
    ),
}
