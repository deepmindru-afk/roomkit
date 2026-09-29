"""Data set apart inside text a model reads.

A tool's result or a worker's output is external data. Framed in a tagged
block, it must not be able to end the block early: what followed would read
as the text around it, prompt or instruction.
"""

from __future__ import annotations

import re


def fence(tag: str, text: str) -> str:
    """*text* inside ``<tag>`` … ``</tag>``, with no closing tag of its own.

    Any closing tag of that name in *text*, in any case and with any spacing
    (``</TOOL_RESULT >``, ``< / tool_result>``), is neutralised, so the data
    cannot close the block.
    """
    closing = re.compile(rf"<\s*/\s*{re.escape(tag)}\s*>", re.IGNORECASE)
    body = closing.sub(f"</{tag}_>", text)
    return f"<{tag}>\n{body}\n</{tag}>"
