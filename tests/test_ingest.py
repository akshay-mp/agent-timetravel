"""Unit tests for ``agent_timetravel.ingest`` — pure protobuf → Span decoding.

These tests do **not** spin up a server: they call the pure decode functions
directly on constructed proto messages. Fidelity (the Phase 1 exit criterion)
is asserted by hashing the raw payload and comparing against the source.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2 as ts
from opentelemetry.proto.common.v1 import common_pb2 as c

from agent_timetravel.enums import SpanKind, SpanStatus
from agent_timetravel.ingest import (
    IngestError,
    anyvalue_to_python,
    attrs_to_dict,
    decode_export_request,
    decode_export_request_json,
    parse_openinference_messages,
    spans_from_request,
)
from agent_timetravel.models import hash_payload

# Deterministic trace/span byte ids. OTel uses raw bytes; the canonical hex-id
# for an OTel trace is 32 hex chars; a span id is 16 hex chars.
_TRACE_ID_BYTES = bytes.fromhex("abcdef1234567890abcdef1234567890")
_AGENT_SPAN_ID = bytes.fromhex("1111111111111111")
_LLM_SPAN_ID = bytes.fromhex("2222222222222222")
_TOOL_SPAN_ID = bytes.fromhex("3333333333333333")


# --- helpers ---------------------------------------------------------------


def _kv(key: str, **oneof: object) -> c.KeyValue:
    """Build a ``KeyValue`` setting the first oneof option in ``oneof``.

    Scalar oneofs (``string_value``, ``int_value``, etc.) accept direct
    ``setattr``. Nested message oneofs (``array_value`` / ``kvlist_value``)
    must be merged via ``CopyFrom`` since protobuf forbids field assignment to
    message subfields. We dispatch by inspecting the sub-attribute's type.
    """
    kv = c.KeyValue(key=key)
    value = c.AnyValue()
    for field, v in oneof.items():
        sub = getattr(value, field)
        if hasattr(sub, "CopyFrom"):
            # Nested message field — CopyFrom reqs the proto type, not AnyValue.
            sub.CopyFrom(v)  # type: ignore[arg-type]
        else:
            setattr(value, field, v)
    kv.value.CopyFrom(value)
    return kv


def _build_three_span_request() -> ts.ExportTraceServiceRequest:
    """Build a request with an agent → llm → tool chain under one resource.

    Resource carries ``service.name``, exercise that resource attrs merge into
    each span's ``raw_attributes``. The LLM span has GenAI usage tokens, the
    agent span uses OpenInference kind tagging, the tool span uses ``tool.*``.
    """
    req = ts.ExportTraceServiceRequest()
    rs = req.resource_spans.add()
    rs.resource.attributes.extend(
        [
            _kv("service.name", string_value="demo-agent"),
            _kv("telemetry.sdk.language", string_value="python"),
        ]
    )
    ss = rs.scope_spans.add()

    # Agent span — classified via openinference.span.kind.
    agent = ss.spans.add()
    agent.trace_id = _TRACE_ID_BYTES
    agent.span_id = _AGENT_SPAN_ID
    # parent_span_id stays empty → None on the TimeTravel model.
    agent.name = "ADK.CustomerCareAgent"
    agent.start_time_unix_nano = 1_700_000_000_000_000_000
    agent.end_time_unix_nano = 1_700_000_005_000_000_000
    agent.status.code = 1  # OK
    agent.attributes.extend([_kv("openinference.span.kind", string_value="AGENT")])

    # LLM span — child of agent. Carries tokens and messages.
    llm = ss.spans.add()
    llm.trace_id = _TRACE_ID_BYTES
    llm.span_id = _LLM_SPAN_ID
    llm.parent_span_id = _AGENT_SPAN_ID
    llm.name = "chatchat.completions.openai"
    llm.start_time_unix_nano = 1_700_000_001_000_000_000
    llm.end_time_unix_nano = 1_700_000_002_000_000_000
    llm.status.code = 1
    llm.attributes.extend(
        [
            _kv("gen_ai.system", string_value="openai"),
            _kv("gen_ai.request.model", string_value="gpt-4o"),
            _kv("gen_ai.response.model", string_value="gpt-4o-2024-08-06"),
            _kv("gen_ai.usage.prompt_tokens", int_value=42),
            _kv("gen_ai.usage.completion_tokens", int_value=7),
            _kv(
                "llm.input_messages",
                array_value=c.ArrayValue(
                    values=[
                        c.AnyValue(kvlist_value=c.KeyValueList(
                            values=[
                                _kv("role", string_value="user"),
                                _kv("content", string_value="hi"),
                            ]
                        ))
                    ]
                ),
            ),
        ]
    )

    # Tool span — child of agent, classified via ``tool.name`` key.
    tool = ss.spans.add()
    tool.trace_id = _TRACE_ID_BYTES
    tool.span_id = _TOOL_SPAN_ID
    tool.parent_span_id = _AGENT_SPAN_ID
    tool.name = "tool.search"
    tool.start_time_unix_nano = 1_700_000_003_000_000_000
    tool.end_time_unix_nano = 1_700_000_004_000_000_000
    tool.status.code = 2  # ERROR
    tool.status.message = "boom"
    tool.attributes.extend(
        [
            _kv("tool.name", string_value="search_products"),
            _kv("tool.output", string_value="[]"),
        ]
    )

    return req


# --- decode_export_request -------------------------------------------------


class TestDecodeProtobuf:
    def test_decodes_valid_protobuf_round_trip(self) -> None:
        req = _build_three_span_request()
        blob = req.SerializeToString()
        out = decode_export_request(blob)
        assert len(out.resource_spans) == 1
        assert len(out.resource_spans[0].scope_spans[0].spans) == 3

    def test_malformed_bytes_raise_ingest_error(self) -> None:
        with pytest.raises(IngestError):
            decode_export_request(b"not-protobuf-at-all")


class TestDecodeJSON:
    def test_decodes_json_via_google_json_format(self) -> None:
        from google.protobuf import json_format

        req = _build_three_span_request()
        json_str = json_format.MessageToJson(req)
        out = decode_export_request_json(json_str)
        assert len(out.resource_spans[0].scope_spans[0].spans) == 3

    def test_bad_json_raises_ingest_error(self) -> None:
        with pytest.raises(IngestError):
            decode_export_request_json("{not valid json")


# --- spans_from_request ----------------------------------------------------


class TestSpansFromRequest:
    @pytest.fixture
    def spans(self) -> list:
        return spans_from_request(_build_three_span_request())

    def test_three_spans_extracted(self, spans: list) -> None:
        assert len(spans) == 3

    def test_trace_and_span_ids_are_lower_hex(self, spans: list) -> None:
        agent = next(s for s in spans if s.name.startswith("ADK"))
        assert agent.trace_id == _TRACE_ID_BYTES.hex()
        assert agent.span_id == _AGENT_SPAN_ID.hex()
        assert agent.parent_span_id is None

    def test_parent_linking_round_trips(self, spans: list) -> None:
        agent_id = _AGENT_SPAN_ID.hex()
        children = [s for s in spans if s.parent_span_id == agent_id]
        assert {c.span_id for c in children} == {_LLM_SPAN_ID.hex(), _TOOL_SPAN_ID.hex()}

    def test_resource_attributes_merge_into_each_span(self, spans: list) -> None:
        for s in spans:
            assert s.raw_attributes["service.name"] == "demo-agent"
            assert s.raw_attributes["telemetry.sdk.language"] == "python"

    def test_classification_correct(self, spans: list) -> None:
        kinds = {s.name: s.kind for s in spans}
        assert kinds["ADK.CustomerCareAgent"] == SpanKind.AGENT
        # The LLM span has gen_ai.usage.* — wins over openinference absence.
        llm = next(s for s in spans if s.span_id == _LLM_SPAN_ID.hex())
        assert llm.kind == SpanKind.LLM
        assert kinds["tool.search"] == SpanKind.TOOL

    def test_status_codes_map_correctly(self, spans: list) -> None:
        tool = next(s for s in spans if s.name == "tool.search")
        assert tool.status == SpanStatus.ERROR
        assert tool.status_message == "boom"
        agent = next(s for s in spans if s.name.startswith("ADK"))
        assert agent.status == SpanStatus.OK

    def test_nanoseconds_convert_to_iso_utc(self, spans: list) -> None:
        agent = next(s for s in spans if s.name.startswith("ADK"))
        # 1_700_000_000s = 2023-11-14T22:13:20Z
        dt = datetime.fromisoformat(agent.start_time)
        assert dt.tzinfo is not None
        assert dt.tzinfo.utcoffset(dt) == datetime.now(tz=UTC).utcoffset()
        assert dt.year == 2023

    def test_tokens_total_summed_when_missing(self, spans: list) -> None:
        llm = next(s for s in spans if s.span_id == _LLM_SPAN_ID.hex())
        assert llm.prompt_tokens == 42
        assert llm.completion_tokens == 7
        assert llm.total_tokens == 49  # came through verbatim

    def test_model_name_prefers_response_model(self, spans: list) -> None:
        llm = next(s for s in spans if s.span_id == _LLM_SPAN_ID.hex())
        assert llm.model_name == "gpt-4o-2024-08-06"


# --- Phase 1 exit criterion: prompt fidelity ------------------------------


class TestFidelity:
    """The golden exit criterion — hash(raw_attributes['payload']) matches source."""

    def test_messages_hash_matches_source_payload(self) -> None:
        req = _build_three_span_request()
        spans = spans_from_request(req)
        llm = next(s for s in spans if s.span_id == _LLM_SPAN_ID.hex())

        # The source payload — what an instrumented agent emitted.
        llm_proto = next(
            s for s in req.resource_spans[0].scope_spans[0].spans if s.span_id == _LLM_SPAN_ID
        )
        source_payload = attrs_to_dict(list(llm_proto.attributes))["llm.input_messages"]

        assert llm.messages_hash == hash_payload(source_payload)

    def test_flat_adk_messages_hash_compacts_text_and_tool_calls(self) -> None:
        req = _build_three_span_request()
        llm_proto = next(
            s
            for s in req.resource_spans[0].scope_spans[0].spans
            if s.span_id == _LLM_SPAN_ID
        )
        llm_proto.ClearField("attributes")
        flat = {
            "openinference.span.kind": "LLM",
            "llm.input_messages.4.message.role": "tool",
            "llm.input_messages.4.message.content": "{\"temp\": 72}",
            "llm.input_messages.4.message.tool_call_id": "call_1",
            "llm.input_messages.4.message.name": "weather",
            "llm.input_messages.2.message.role": "assistant",
            "llm.input_messages.2.message.contents.1.message_content.type": "tool_use",
            "llm.input_messages.2.message.contents.1.tool_call.id": "call_1",
            "llm.input_messages.2.message.contents.1.tool_call.function.name": "weather",
            "llm.input_messages.2.message.contents.1.tool_call.function.arguments":
                '{"city":"Boston"}',
            "llm.input_messages.2.message.tool_calls.0.tool_call.id": "call_1",
            "llm.input_messages.2.message.tool_calls.0.tool_call.function.name": "weather",
            "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments":
                '{"city":"Boston"}',
            "llm.input_messages.2.message.contents.0.message_content.type": "text",
            "llm.input_messages.2.message.contents.0.message_content.text": "Checking",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.contents.0.message_content.type": "text",
            "llm.input_messages.0.message.contents.0.message_content.text": "Weather?",
        }
        llm_proto.attributes.extend(_kv(key, string_value=value) for key, value in flat.items())

        expected = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "Checking",
                "tool_calls": [
                    {
                        "name": "weather",
                        "args": {"city": "Boston"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"temp":72}',
                "tool_call_id": "call_1",
                "name": "weather",
            },
        ]
        attrs = attrs_to_dict(llm_proto.attributes)
        assert parse_openinference_messages(attrs) == expected
        span = next(s for s in spans_from_request(req) if s.span_id == _LLM_SPAN_ID.hex())
        assert span.messages_hash == hash_payload(expected)


# --- anyvalue_to_python ----------------------------------------------------


class TestAnyValueUnwrap:
    def test_string(self) -> None:
        av = c.AnyValue(string_value="hi")
        assert anyvalue_to_python(av) == "hi"

    def test_int(self) -> None:
        av = c.AnyValue(int_value=42)
        assert anyvalue_to_python(av) == 42

    def test_bool(self) -> None:
        av = c.AnyValue(bool_value=False)
        assert anyvalue_to_python(av) is False

    def test_double(self) -> None:
        av = c.AnyValue(double_value=1.5)
        assert anyvalue_to_python(av) == 1.5

    def test_bytes_as_int_list(self) -> None:
        av = c.AnyValue(bytes_value=b"\x01\x02")
        assert anyvalue_to_python(av) == [1, 2]

    def test_array(self) -> None:
        av = c.AnyValue(
            array_value=c.ArrayValue(
                values=[c.AnyValue(string_value="a"), c.AnyValue(int_value=1)]
            )
        )
        assert anyvalue_to_python(av) == ["a", 1]

    def test_kvlist(self) -> None:
        av = c.AnyValue(
            kvlist_value=c.KeyValueList(
                values=[
                    c.KeyValue(key="k1", value=c.AnyValue(string_value="v1")),
                ]
            )
        )
        assert anyvalue_to_python(av) == {"k1": "v1"}

    def test_unset_returns_sentinel(self) -> None:
        av = c.AnyValue()
        result = anyvalue_to_python(av)
        assert isinstance(result, str)
        assert result.startswith("<unset")


# --- attrs_to_dict ---------------------------------------------------------


class TestAttrsToDict:
    def test_empty(self) -> None:
        assert attrs_to_dict([]) == {}

    def test_multiple_keys_preserved(self) -> None:
        kvs = [_kv("a", string_value="1"), _kv("b", int_value=2)]
        assert attrs_to_dict(kvs) == {"a": "1", "b": 2}


# --- integration-ish: error categories ------------------------------------


class TestEmptyAndDegenerateRequests:
    def test_empty_request_yields_no_spans(self) -> None:
        assert spans_from_request(ts.ExportTraceServiceRequest()) == []

    def test_resource_with_no_scope_yields_no_spans(self) -> None:
        req = ts.ExportTraceServiceRequest()
        req.resource_spans.add().resource.attributes.extend(
            [_kv("service.name", string_value="empty")]
        )
        assert spans_from_request(req) == []


# Avoid "unused import" if uuid4 is later moved out — this anchor keeps the
# test module stable under the strict unused-symbol linters.
_ = uuid4()
