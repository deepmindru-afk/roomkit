"""Whether a provider talks to its vendor's own endpoint (RFC §6.7).

A provider knows its vendor's rules (the tool names it accepts, the
capabilities its catalogue states, the defaults a modern model needs) on the
vendor's own endpoint only: behind another ``base_url`` the server decides.
The vendor's own endpoint is the SDK's default, or one of the vendor's official
URLs written out: a configuration naming ``https://api.openai.com/v1`` talks to
OpenAI, not to a proxy, and gets OpenAI's rules.
"""

from __future__ import annotations

from urllib.parse import urlsplit

OPENAI_BASE_URL = "https://api.openai.com/v1"
"""OpenAI's REST endpoint, the SDK's default."""

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
"""Anthropic's API, the SDK's default."""

DEEPSEEK_BASE_URLS = ("https://api.deepseek.com/v1", "https://api.deepseek.com")
"""DeepSeek's endpoint, under the two bases its documentation gives."""


def is_vendor_endpoint(base_url: str | None, *official: str) -> bool:
    """Whether *base_url* is the vendor's own endpoint: none given (the SDK's
    default), or one of its *official* URLs, whatever its case of scheme and
    host and its trailing slash."""
    if base_url is None:
        return True
    return _comparable(base_url) in {_comparable(url) for url in official}


def _comparable(url: str) -> tuple[str, str, str]:
    """*url* reduced to what names an endpoint: scheme, host and path."""
    parts = urlsplit(url.strip())
    return parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/")
