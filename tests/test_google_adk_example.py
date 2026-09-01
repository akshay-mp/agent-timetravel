"""Focused, no-network checks for the Google ADK example."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

types = pytest.importorskip("google.genai.types")
pytest.importorskip("google.adk")
pytest.importorskip("openai")

_APP_PATH = Path(__file__).parents[1] / "examples/google_adk_demo/app.py"
_APP_SPEC = importlib.util.spec_from_file_location("google_adk_demo_app", _APP_PATH)
if _APP_SPEC is None or _APP_SPEC.loader is None:
    raise AssertionError("could not load Google ADK example module")
app = importlib.util.module_from_spec(_APP_SPEC)
sys.modules[_APP_SPEC.name] = app
_APP_SPEC.loader.exec_module(app)


def test_schema_types_are_lowercase_recursively() -> None:
    """Nested ADK schema enums become OpenAI-compatible JSON Schema types."""
    schema = {
        "type": types.Type.OBJECT,
        "properties": {
            "city": {"type": types.Type.STRING},
            "forecast": {
                "type": types.Type.ARRAY,
                "items": {
                    "type": types.Type.OBJECT,
                    "properties": {"temperature": {"type": types.Type.NUMBER}},
                },
            },
        },
    }

    converted = app._normalize_schema_types(schema)

    assert converted == {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "forecast": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"temperature": {"type": "number"}},
                },
            },
        },
    }
    assert schema["type"] is types.Type.OBJECT


def test_schema_type_unions_are_lowercase_and_preserve_other_values() -> None:
    """ADK schema type unions normalize without contacting a service."""
    object_value = object()
    assert app._normalize_schema_type(
        [types.Type.STRING, "NUMBER", object_value]
    ) == ["string", "number", object_value]
    assert app._normalize_schema_type((types.Type.BOOLEAN, "NULL")) == (
        "boolean",
        "null",
    )


def test_declarations_to_tools_normalizes_adk_parameter_schema() -> None:
    """The tool conversion applies normalization to ADK declarations."""
    declaration = SimpleNamespace(
        name="weather",
        description="Get weather",
        parameters_json_schema={
            "type": types.Type.OBJECT,
            "properties": {"city": {"type": types.Type.STRING}},
        },
    )
    request = SimpleNamespace(
        config=SimpleNamespace(
            tools=[SimpleNamespace(function_declarations=[declaration])]
        )
    )

    tools = app._declarations_to_tools(request)

    assert tools is not None
    assert tools[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }


def test_example_import_registers_adk_agent() -> None:
    """Importing the example registers its ADK agent without making a request."""
    definition = app.timetravel.get("ask")

    assert definition is not None
    assert definition.framework == "adk"
    assert definition.func is app.ask
