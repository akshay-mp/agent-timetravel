"""OTLP protobuf decoding — pure, side-effect free.

This module is the seam between the OTel wire format and TimeTravel's ``Span``
domain model. It contains **no I/O** so it can be unit-tested directly:

    spans = decode_export_request(protobuf_bytes)

The receiver (``receiver.py``) calls these pure functions inside a FastAPI
handler, and persists results via ``TraceStore``.

We accept the OTLP/HTTP shape from
`opentelemetry.proto.collector.trace.v1.ExportTraceServiceRequest`. Both
binary protobuf ("application/x-protobuf") and canonical JSON
("application/json") are supported — the OTel Python SDK ships both, and
OpenInference exporters may use either.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from google.protobuf import json_format
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2 as ts
from opentelemetry.proto.common.v1 import common_pb2
from opentelemetry.proto.trace.v1 import trace_pb2 as tpb

from agent_timetravel.classify import classify_span
from agent_timetravel.enums import SpanKind, SpanStatus
from agent_timetravel.models import Span, hash_payload

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Map from OTel ``StatusCode`` int → TimeTravel ``SpanStatus``.
#:
#: OTel values: 0=UNSET, 1=OK, 2=ERROR. See
#: ``opentelemetry.proto.trace.v1.Status``.
_STATUS_BY_CODE: dict[int, SpanStatus] = {
    0: SpanStatus.UNSET,
    1: SpanStatus.OK,
    2: SpanStatus.ERROR,
}

#: Attribute keys we promote into typed, queryable Span fields. Everything
#: else goes verbatim into ``raw_attributes``.
_PROMOTED = {
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.usage.prompt_tokens",
    "gen_ai.usage.completion_tokens",
    "gen_ai.usage.total_tokens",
}

_FLAT_MESSAGE_RE = re.compile(r"^(?P<prefix>.+)\.(?P<index>\d+)\.message\.(?P<field>.+)$")
_FLAT_CONTENT_RE = re.compile(
    r"^contents\.(?P<index>\d+)\.(?P<field>message_content\.(?:text|type)|tool_call\..+)$"
)
_FLAT_TOOL_CALL_RE = re.compile(
    r"^tool_calls\.(?P<index>\d+)\.(?P<field>tool_call\..+)$"
)


class IngestError(ValueError):
    """Raised when the OTLP payload cannot be decoded or is malformed."""


def decode_export_request(blob: bytes) -> ts.ExportTraceServiceRequest:
    """Parse raw OTLP/HTTP protobuf bytes into an ``ExportTraceServiceRequest``.

    Raises :class:`IngestError` if the bytes are not valid protobuf.
    """
    # Generated proto modules expose messages at runtime; pylint can't see them.
    req = ts.ExportTraceServiceRequest()  # pylint: disable=no-member
    try:
        req.ParseFromString(blob)
    except DecodeError as exc:  # pragma: no cover - defensive; bad framing
        raise IngestError(f"failed to parse ExportTraceServiceRequest: {exc}") from exc
    return req


def decode_export_request_json(blob: bytes | str) -> ts.ExportTraceServiceRequest:
    """Parse JSON-encoded OTLP into an ``ExportTraceServiceRequest``.

    The OTLP/HTTP JSON codec is field-number agnostic — keys are camelCase
    field names per ``trace_service.proto``. ``json_format.Parse`` is strict
    and rejects unknown fields, which gives us a clean validation surface.
    """
    req = ts.ExportTraceServiceRequest()  # pylint: disable=no-member
    try:
        json_format.Parse(_as_text(blob), req, ignore_unknown_fields=False)
    except json_format.ParseError as exc:
        raise IngestError(f"failed to parse JSON OTLP request: {exc}") from exc
    return req


def _as_text(blob: bytes | str) -> str:
    return blob.decode("utf-8") if isinstance(blob, bytes) else blob


def spans_from_request(req: ts.ExportTraceServiceRequest) -> list[Span]:
    """Flatten an ``ExportTraceServiceRequest`` into TimeTravel ``Span``s.

    Walks ``resource_spans[].scope_spans[].spans[]``, decodes attributes and
    timestamps, classifies kind, and computes typed fields/messages-hash.
    Resource attributes (e.g. ``service.name``) are merged into each span's
    ``raw_attributes`` so the workspace context is preserved per-span.
    """
    out: list[Span] = []
    for rs in req.resource_spans:
        resource_attrs = attrs_to_dict(rs.resource.attributes)
        for scope_spans in rs.scope_spans:
            for sp in scope_spans.spans:
                out.append(_span_from_proto(sp, resource_attrs))
    return out


def _span_from_proto(sp: tpb.Span, resource_attrs: dict[str, Any]) -> Span:
    """Translate one OTel ``Span`` proto into a TimeTravel ``Span``.

    The raw attributes dict is the union of resource + span attributes.
    Typed fields (model, tokens, hashes) are *derived* from that union so a
    SQL filter and a Python-side read of ``raw_attributes`` can never diverge.
    """
    raw = {**resource_attrs, **attrs_to_dict(sp.attributes)}
    kind = classify_span(sp.name, raw)
    status = _status_from_proto(sp.status.code)

    model_name = _coerce_model_name(raw)
    prompt, completion, total = _token_triple(raw)

    messages_hash, tools_hash = _hashes_for_kind(kind, raw)

    return Span(
        trace_id=_hex(sp.trace_id),
        span_id=_hex(sp.span_id),
        parent_span_id=_hex(sp.parent_span_id) or None,
        name=sp.name,
        kind=kind,
        start_time=_nanos_to_iso(sp.start_time_unix_nano),
        end_time=_nanos_to_iso(sp.end_time_unix_nano),
        status=status,
        status_message=sp.status.message or None,
        model_name=model_name,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        messages_hash=messages_hash,
        tools_hash=tools_hash,
        raw_attributes=raw,
    )


def _status_from_proto(code: int) -> SpanStatus:
    """Map an OTel ``StatusCode`` int → ``SpanStatus`` (defensive on bad ints)."""
    return _STATUS_BY_CODE.get(code, SpanStatus.UNSET)


def _coerce_model_name(raw: dict[str, Any]) -> str | None:
    """Prefer response.model (what actually served), fall back to request.model."""
    return (
        _as_str(raw.get("gen_ai.response.model"))
        or _as_str(raw.get("gen_ai.request.model"))
        or None
    )


def _token_triple(raw: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """Extract (prompt, completion, total) tokens from GenAI semconv attributes."""
    prompt = _as_int(raw.get("gen_ai.usage.prompt_tokens"))
    completion = _as_int(raw.get("gen_ai.usage.completion_tokens"))
    total = _as_int(raw.get("gen_ai.usage.total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return prompt, completion, total


def _hashes_for_kind(kind: SpanKind, raw: dict[str, Any]) -> tuple[str | None, str | None]:
    """Compute ``messages_hash`` / ``tools_hash`` for LLM-classified spans.

    Frozen replay (P3) matches on ``model + messages_hash + tools_hash``. We
    compute these at ingest time so the replay cursor can do a single SQL
    equality on an indexed column instead of re-hashing the raw payload.

    Keys follow OpenInference convention (``llm.input_messages``, etc.) and
    GenAI 1.0 semconv (``gen_ai.prompt``/``gen_ai.completion``). Anything we
    can't find is left as ``None``; matching tolerates missing hashes.
    """
    if kind != SpanKind.LLM:
        return None, None

    messages_payload = _first_present(
        raw,
        "llm.input_messages",
        "gen_ai.prompt",
        "gen_ai.input.messages",
    )
    if messages_payload is None:
        flattened_messages = parse_openinference_messages(raw)
        if flattened_messages:
            messages_payload = flattened_messages
    tools_payload = _first_present(
        raw,
        "llm.tools",
        "gen_ai.tools",
    )
    messages_hash = hash_payload(messages_payload) if messages_payload is not None else None
    tools_hash = hash_payload(tools_payload) if tools_payload is not None else None
    return messages_hash, tools_hash


def parse_openinference_messages(
    attributes: dict[str, Any],
    *,
    prefix: str = "llm.input_messages",
) -> list[dict[str, Any]]:
    """Rebuild compact messages from OpenInference's flattened attributes.

    The result deliberately matches the small message shape used by the ADK
    interceptor: ``role`` + flattened text ``content`` and, when present,
    native ``tool_calls`` entries with ``name``/``args``/``id``/``type``.
    Both conventional ``message.tool_calls`` and ordered ``message.contents``
    tool-use items are accepted. Malformed or unsupported fields are ignored.
    """
    messages: dict[int, dict[str, Any]] = {}
    for key, value in attributes.items():
        match = _FLAT_MESSAGE_RE.match(key)
        if match is None or match.group("prefix") != prefix:
            continue
        index = int(match.group("index"))
        messages.setdefault(index, {})[match.group("field")] = value

    out: list[dict[str, Any]] = []
    for index in sorted(messages):
        fields = messages[index]
        message: dict[str, Any] = {
            "role": _as_str(fields.get("role")) or "user",
            "content": _as_message_content(fields.get("content")),
        }
        if "name" in fields:
            message["name"] = str(fields["name"])
        if "tool_call_id" in fields:
            message["tool_call_id"] = str(fields["tool_call_id"])
        if message["role"] == "model":
            message["role"] = "assistant"
        if message["role"] == "tool":
            message["content"] = _canonical_json_content(message["content"])

        calls: list[dict[str, Any]] = []
        for call_index in _flat_indexes(fields, _FLAT_TOOL_CALL_RE):
            call = _tool_call_from_fields(_flat_group(fields, _FLAT_TOOL_CALL_RE, call_index))
            _append_tool_call(calls, call)

        direct_call = _tool_call_from_fields(
            {
                "tool_call.function.name": fields.get("function_call_name"),
                "tool_call.function.arguments": fields.get(
                    "function_call_arguments_json", fields.get("function_call_arguments")
                ),
            }
        )
        _append_tool_call(calls, direct_call)

        content_text: list[str] = []
        for content_index in _flat_indexes(fields, _FLAT_CONTENT_RE):
            content_fields = _flat_group(fields, _FLAT_CONTENT_RE, content_index)
            content_type = _as_str(content_fields.get("message_content.type"))
            text = content_fields.get("message_content.text")
            if content_type in (None, "text") and text is not None:
                content_text.append(str(text))
            if content_type == "tool_use":
                _append_tool_call(calls, _tool_call_from_fields(content_fields))

        if "content" not in fields and content_text:
            message["content"] = "\n".join(content_text)
        if calls:
            message["tool_calls"] = calls
        out.append(message)
    return out


def _flat_indexes(fields: dict[str, Any], pattern: re.Pattern[str]) -> list[int]:
    indexes: set[int] = set()
    for key in fields:
        match = pattern.match(key)
        if match is not None:
            indexes.add(int(match.group("index")))
    return sorted(indexes)


def _flat_group(
    fields: dict[str, Any], pattern: re.Pattern[str], index: int
) -> dict[str, Any]:
    prefix = f"tool_calls.{index}." if pattern is _FLAT_TOOL_CALL_RE else f"contents.{index}."
    return {
        match.group("field"): value
        for key, value in fields.items()
        if key.startswith(prefix) and (match := pattern.match(key)) is not None
    }


def _tool_call_from_fields(fields: dict[str, Any]) -> dict[str, Any] | None:
    name = _as_str(fields.get("tool_call.function.name"))
    if name is None:
        return None
    arguments = fields.get("tool_call.function.arguments")
    if arguments is None:
        arguments = fields.get("tool_call.function.arguments_json")
    return {
        "name": name,
        "args": _tool_call_args(arguments),
        "id": _as_str(fields.get("tool_call.id")) or "",
        "type": "tool_call",
    }


def _tool_call_args(value: Any) -> dict[str, Any]:  # noqa: ANN401
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _append_tool_call(calls: list[dict[str, Any]], call: dict[str, Any] | None) -> None:
    if call is not None and call not in calls:
        calls.append(call)


def _as_message_content(value: Any) -> str:  # noqa: ANN401
    return "" if value is None else str(value)


def _canonical_json_content(value: str) -> str:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return value
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), default=str)


def _first_present(d: dict[str, Any], *keys: str) -> Any | None:  # noqa: ANN401
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _as_str(v: Any) -> str | None:  # noqa: ANN401
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _as_int(v: Any) -> int | None:  # noqa: ANN401
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _hex(b: bytes) -> str:
    """Lowercase hex of an OTel ``trace_id``/``span_id`` byte field.

    Empty bytes (no parent_span_id on root spans) → empty string.
    """
    if not b:
        return ""
    return b.hex()


def _nanos_to_iso(nanos: int) -> str:
    """Convert OTel unix-nanoseconds to ISO-8601 UTC string.

    OTel uses uint64 nanos since epoch. We never store zero — a missing
    start/end is preserved as an "epoch" timestamp for round-trip safety
    rather than crashing.
    """
    seconds, partial = divmod(int(nanos), 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=partial // 1000)
    return dt.isoformat()


def attrs_to_dict(kvs: Iterable[common_pb2.KeyValue]) -> dict[str, Any]:
    """Convert a repeated ``KeyValue`` proto into a plain Python dict.

    The OTel ``AnyValue`` oneof is fully unwrapped: primitive scalars come out
    as their native type, ``bytes_value`` as a list of ints (so JSON-safe),
    repeated data as lists, and nested ``kvlist_value`` as nested dicts. This
    is deliberately faithful — it is what we hash against at ingest time.
    """
    out: dict[str, Any] = {}
    for kv in kvs:
        out[kv.key] = anyvalue_to_python(kv.value)
    return out


def anyvalue_to_python(av: common_pb2.AnyValue) -> Any:  # noqa: ANN401
    """Unwrap an OTel ``AnyValue`` oneof to a native Python value.

    Uses ``WhichOneof`` rather than ``HasField`` because the oneof ("value")
    may be set to the default-but-present variant (e.g. ``bool_value=False``).
    Unknown variants serialize as a string tag + raw repr, which still
    round-trips for hash purposes.
    """
    which = av.WhichOneof("value")
    if which == "string_value":
        return av.string_value
    if which == "bool_value":
        return av.bool_value
    if which == "int_value":
        return av.int_value
    if which == "double_value":
        return av.double_value
    if which == "bytes_value":
        return list(av.bytes_value)
    if which == "array_value":
        return [anyvalue_to_python(v) for v in av.array_value.values]
    if which == "kvlist_value":
        return {kv.key: anyvalue_to_python(kv.value) for kv in av.kvlist_value.values}
    # Unknown / unset oneof. Sentry value so we never silently drop data.
    return f"<unset:{which or 'none'}>"


__all__ = [
    "IngestError",
    "anyvalue_to_python",
    "attrs_to_dict",
    "decode_export_request",
    "decode_export_request_json",
    "parse_openinference_messages",
    "spans_from_request",
]
