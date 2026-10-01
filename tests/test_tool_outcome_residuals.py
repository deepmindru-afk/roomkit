"""Leftovers of the tool-result review that read a result wrong (RMK-305, RMK-300).

A mapping that names a part type among other data is data, not a part (RFC
§21.4); a tool's search tags survive its extraction; a read of a stored result
that is not there is a refusal, not a result.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from roomkit.channels._tool_eviction import ToolEviction
from roomkit.channels._tool_registry import schema_tool
from roomkit.core.exceptions import ChannelRefusalError
from roomkit.tools.compose import extract_tools
from roomkit.tools.result import as_tool_result


class TestDataIsNotAPart:
    def test_a_mapping_with_a_part_type_and_more_fields_is_json(self) -> None:
        chunk = [{"type": "text", "text": "chunk", "page": 2}]

        result = as_tool_result(chunk)

        assert json.loads(result) == chunk  # the page survives

    def test_an_image_shaped_record_with_extra_fields_is_json(self) -> None:
        record = [{"type": "image", "url": "https://x/y.png", "caption": "a cat", "id": 7}]

        assert json.loads(as_tool_result(record)) == record

    def test_a_part_in_its_exact_shape_is_still_a_part(self) -> None:
        parts = as_tool_result([{"type": "image", "url": "data:,", "mime_type": "image/png"}])

        assert isinstance(parts, list)


class _Tagged:
    @property
    def definition(self) -> dict[str, Any]:
        return {"name": "weather", "description": "Weather", "tags": ["météo", "forecast"]}

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        return "{}"


class TestTagsSurvive:
    def test_extracting_a_tool_object_keeps_its_tags(self) -> None:
        definitions, _ = extract_tools([_Tagged()])

        assert definitions[0].tags == ["météo", "forecast"]

    def test_extracting_a_schema_keeps_its_tags(self) -> None:
        definitions, _ = extract_tools([{"name": "lookup", "tags": ["crm"]}])

        assert definitions[0].tags == ["crm"]

    def test_a_channel_tool_schema_keeps_its_tags(self) -> None:
        assert schema_tool({"name": "lookup", "tags": ["crm"]}).tags == ["crm"]


def test_reading_a_stored_result_that_is_not_there_is_a_refusal() -> None:
    eviction = ToolEviction(threshold_tokens=5000)

    with pytest.raises(ChannelRefusalError) as refused:
        eviction.handle_read({"result_id": "missing"})

    assert "not found" in json.loads(refused.value.message)["error"]
