"""The llama.cpp build the llamacpp provider downloads — generated, do not edit.

Regenerate with ``uv run python scripts/update_llamacpp_build.py``.
"""

# ruff: noqa: E501

from __future__ import annotations

BUILD = "b11160"

# variant -> ((archive, sha256), ...): the build, then its CUDA runtime if any.
ASSETS: dict[str, tuple[tuple[str, str], ...]] = {
    "linux-x64-cpu": (
        (
            "llama-b11160-bin-ubuntu-x64.tar.gz",
            "48ece24283876fc3401b737724008c03cbc4c7ba335b6c1aa2a7b6ce2d49e435",
        ),
    ),
    "linux-x64-cuda-12": (
        (
            "llama-b11160-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "8ed8d659383b6624ef2bd14ebd6984c8c1b1b2f13e4ba0c5c43f6038ca22591d",
        ),
        (
            "cudart-llama-b11160-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "54776c67e34b536f6123b1a0697931f86f846f9f71baa9d86f550485fbbc9df1",
        ),
    ),
    "linux-x64-cuda-13": (
        (
            "llama-b11160-bin-ubuntu-cuda-13.4-x64.tar.gz",
            "76983c34644683614a0d4983b7948b503dac76bfacc9f96a811c0ca146f103d4",
        ),
        (
            "cudart-llama-b11160-bin-ubuntu-cuda-13.4-x64.tar.gz",
            "14765bd08136c6838fbadf22f3484d2b9ad2be307ae0fa766ec42f4bb100b070",
        ),
    ),
    "linux-x64-vulkan": (
        (
            "llama-b11160-bin-ubuntu-vulkan-x64.tar.gz",
            "4dd1285b8b1554ce4c87904daaf0c9c0c38a5004187604ae2c6e3c5c212cb6f8",
        ),
    ),
    "linux-arm64-cpu": (
        (
            "llama-b11160-bin-ubuntu-arm64.tar.gz",
            "4ffc2f68959e102219eaa8d007c9fd5b6e7d1cb298672c8945ec80b9f57b6076",
        ),
    ),
    "linux-arm64-cuda-13": (
        (
            "llama-b11160-bin-ubuntu-cuda-13.4-arm64.tar.gz",
            "ef9d913faac8439c50249781191121aed06dfee13e8244d7421122fba48d487f",
        ),
        (
            "cudart-llama-b11160-bin-ubuntu-cuda-13.4-arm64.tar.gz",
            "2ccb86558da98dcb45e1210e3e88bada95edbc5c0e9867893c4b0840cf714e90",
        ),
    ),
    "macos-arm64": (
        (
            "llama-b11160-bin-macos-arm64.tar.gz",
            "5679b3e952772a9f9a39f9d42d7f0eb3d4c424103fe56f5516507583a0c6e3fa",
        ),
    ),
    "macos-x64": (
        (
            "llama-b11160-bin-macos-x64.tar.gz",
            "8c9029bb2491c9c39a497bbd3499d38df0b1c134006bc0e3ec6b5b37319955c8",
        ),
    ),
    "windows-x64-cpu": (
        (
            "llama-b11160-bin-win-cpu-x64.zip",
            "b144d125972c57eb30062524269b31bf981dfb81d36fad6a1494e18814a06acc",
        ),
    ),
    "windows-x64-cuda-12": (
        (
            "llama-b11160-bin-win-cuda-12.4-x64.zip",
            "e2bcd71b9a03e4ed7d8bdf77f129eb682d63bbff244ef24fbbe40f1003368763",
        ),
        (
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
            "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
        ),
    ),
    "windows-x64-cuda-13": (
        (
            "llama-b11160-bin-win-cuda-13.4-x64.zip",
            "966b2b052a820d71ba1c2040a73c46afc772659a75395260a583746effb72cff",
        ),
        (
            "cudart-llama-bin-win-cuda-13.4-x64.zip",
            "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
        ),
    ),
    "windows-arm64-cpu": (
        (
            "llama-b11160-bin-win-cpu-arm64.zip",
            "64ae458e26538cc7e85644ca930055fefbab7a41a84108cb2c6fb5800e5979ae",
        ),
    ),
}
