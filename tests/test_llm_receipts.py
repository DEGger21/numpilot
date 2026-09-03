import hashlib
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from numpilot.engines import llm as llm_module
from numpilot.engines.llm import (
    LlmEngine,
    LlmHttpError,
    LlmResponseError,
    LlmTransportError,
)
from numpilot.llm_config import (
    LlmConfigError,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    model_identity_sha256,
)


class _Response:
    def __init__(self, payload, *, status=200, headers=None):
        self._payload = payload
        self._body = io.BytesIO(payload)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def read(self, size=-1):
        return self._body.read(size)

    def readline(self, size=-1):
        return self._body.readline(size)

    def getcode(self):
        return self.status


def _configure(monkeypatch):
    monkeypatch.setenv("NUMPILOT_LLM_BASE_URL", "https://private.example/v1")
    monkeypatch.setenv("NUMPILOT_LLM_AUTH_TOKEN", "top-secret-token")
    monkeypatch.setenv("NUMPILOT_LLM_MODEL", "glm-requested")
    monkeypatch.setenv("NUMPILOT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv(
        "NUMPILOT_LLM_PROTOCOL", OPENAI_CHAT_COMPLETIONS_PROTOCOL
    )
    monkeypatch.delenv("NUMPILOT_LLM_RESPONSE_MODEL_ALIASES", raising=False)
    monkeypatch.delenv("NUMPILOT_LLM_INSECURE", raising=False)


def test_remote_chat_returns_content_and_redacted_audit_receipt(monkeypatch):
    _configure(monkeypatch)
    content = '{"summary":"ok","files":[]}'
    response_body = json.dumps({
        "id": "response-top-secret-token",
        "model": "org/model",
        "created": 1786555200,
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "cached": True,
            "fractional": 1.5,
            "negative": -1,
            "too_large": 2 ** 63,
        },
    }).encode("utf-8")
    captured = {}

    def fake_urlopen(request, *, timeout, context):
        captured["request"] = request
        captured["timeout"] = timeout
        captured["context"] = context
        return _Response(
            response_body,
            headers={"X-Request-Id": "server-request-top-secret-token"},
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    actual, receipt = LlmEngine._chat_with_receipt("system", "user")

    assert actual == content
    assert receipt["schema_version"] == 2
    assert receipt["utc_start"].endswith("Z")
    assert receipt["utc_end"].endswith("Z")
    assert receipt["http_status"] == 200
    assert receipt["requested_model"] == "glm-requested"
    assert receipt["configured_model"] == "glm-requested"
    assert receipt["response_model"] == "org/model"
    assert receipt["response_model_aliases"] == []
    assert receipt["response_model_identity"] == "mismatch"
    assert receipt["protocol"] == OPENAI_CHAT_COMPLETIONS_PROTOCOL
    assert receipt["provider"] == "openai-compatible"
    assert receipt["model_identity_sha256"] == model_identity_sha256(
        protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        provider="openai-compatible",
        configured_model="glm-requested",
        requested_model="glm-requested",
        response_model="org/model",
        response_model_aliases=(),
        response_model_identity="mismatch",
    )
    assert receipt["response_id"] == "response-redacted"
    assert receipt["response_created"] == 1786555200
    assert receipt["request_id"] == "server-request-redacted"
    assert receipt["tls_verification"] == "enabled"
    assert receipt["request_body_sha256"] == hashlib.sha256(
        captured["request"].data
    ).hexdigest()
    assert receipt["response_body_sha256"] == hashlib.sha256(
        response_body
    ).hexdigest()
    assert receipt["content_sha256"] == hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()
    assert "base_url_sha256" not in receipt
    assert receipt["usage"] == {
        "completion_tokens": 7,
        "prompt_tokens": 11,
        "total_tokens": 18,
    }
    serialized = json.dumps(receipt, sort_keys=True)
    assert "top-secret-token" not in serialized
    assert "https://private.example/v1" not in serialized
    assert captured["context"] is None


@pytest.mark.parametrize(
    ("per_call_max_tokens", "expected_max_tokens"),
    ((None, 6144), (1, 1), (4096, 4096), (1_000_000, 1_000_000)),
)
def test_per_call_max_tokens_overrides_environment_without_changing_default(
    monkeypatch, per_call_max_tokens, expected_max_tokens
):
    _configure(monkeypatch)
    monkeypatch.setenv("NUMPILOT_LLM_MAX_TOKENS", "6144")
    response_body = json.dumps({
        "model": "glm-requested",
        "choices": [{"message": {"content": "ok"}}],
    }).encode("utf-8")
    captured = {}

    def fake_urlopen(request, *, timeout, context):
        del timeout, context
        captured["request_data"] = json.loads(request.data.decode("utf-8"))
        return _Response(response_body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    kwargs = (
        {}
        if per_call_max_tokens is None
        else {"max_tokens": per_call_max_tokens}
    )

    content, _receipt = LlmEngine._chat_with_receipt(
        "system", "user", **kwargs
    )

    assert content == "ok"
    assert captured["request_data"]["max_tokens"] == expected_max_tokens


@pytest.mark.parametrize(
    "invalid_max_tokens",
    (True, False, 0, -1, 1_000_001, 1.0, "4096"),
)
def test_invalid_per_call_max_tokens_fails_before_transport(
    monkeypatch, invalid_max_tokens
):
    _configure(monkeypatch)

    def forbidden_urlopen(*_args, **_kwargs):
        raise AssertionError("invalid per-call max_tokens reached transport")

    monkeypatch.setattr("urllib.request.urlopen", forbidden_urlopen)

    with pytest.raises(LlmConfigError, match="per-call LLM max_tokens"):
        LlmEngine._chat_with_receipt(
            "system", "user", max_tokens=invalid_max_tokens
        )


@pytest.mark.parametrize(
    ("thinking_mode", "expected"),
    ((None, None), ("enabled", {"type": "enabled"}),
     ("disabled", {"type": "disabled"})),
)
def test_per_call_thinking_mode_is_explicit_and_bounded(
    monkeypatch, thinking_mode, expected
):
    _configure(monkeypatch)
    response_body = json.dumps({
        "model": "glm-requested",
        "choices": [{"message": {"content": "ok"}}],
    }).encode("utf-8")
    captured = {}

    def fake_urlopen(request, *, timeout, context):
        del timeout, context
        captured["request_data"] = json.loads(request.data.decode("utf-8"))
        return _Response(response_body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    kwargs = {} if thinking_mode is None else {"thinking_mode": thinking_mode}

    LlmEngine._chat_with_receipt("system", "user", **kwargs)

    assert captured["request_data"].get("thinking") == expected


@pytest.mark.parametrize("thinking_mode", (True, False, "auto", 0, {}))
def test_invalid_per_call_thinking_mode_fails_before_transport(
    monkeypatch, thinking_mode
):
    _configure(monkeypatch)

    def forbidden_urlopen(*_args, **_kwargs):
        raise AssertionError("invalid per-call thinking_mode reached transport")

    monkeypatch.setattr("urllib.request.urlopen", forbidden_urlopen)

    with pytest.raises(LlmConfigError, match="per-call LLM thinking_mode"):
        LlmEngine._chat_with_receipt(
            "system", "user", thinking_mode=thinking_mode
        )


@pytest.mark.parametrize(
    ("reasoning_effort", "expected"),
    ((None, None), ("minimal", "minimal"), ("max", "max")),
)
def test_per_call_reasoning_effort_is_explicit_and_bounded(
    monkeypatch, reasoning_effort, expected
):
    _configure(monkeypatch)
    response_body = json.dumps({
        "model": "glm-requested",
        "choices": [{"message": {"content": "ok"}}],
    }).encode("utf-8")
    captured = {}

    def fake_urlopen(request, *, timeout, context):
        del timeout, context
        captured["request_data"] = json.loads(request.data.decode("utf-8"))
        return _Response(response_body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    kwargs = (
        {} if reasoning_effort is None
        else {"reasoning_effort": reasoning_effort}
    )

    LlmEngine._chat_with_receipt("system", "user", **kwargs)

    assert captured["request_data"].get("reasoning_effort") == expected


@pytest.mark.parametrize("reasoning_effort", (True, "auto", "MINIMAL", 0, {}))
def test_invalid_per_call_reasoning_effort_fails_before_transport(
    monkeypatch, reasoning_effort
):
    _configure(monkeypatch)

    def forbidden_urlopen(*_args, **_kwargs):
        raise AssertionError("invalid reasoning_effort reached transport")

    monkeypatch.setattr("urllib.request.urlopen", forbidden_urlopen)

    with pytest.raises(LlmConfigError, match="reasoning_effort"):
        LlmEngine._chat_with_receipt(
            "system", "user", reasoning_effort=reasoning_effort
        )


def test_streaming_completion_aggregates_content_and_hashes_response_body(
    monkeypatch,
):
    _configure(monkeypatch)
    chunks = (
        {
            "id": "stream-response",
            "model": "glm-requested",
            "created": 1786555200,
            "choices": [{"delta": {
                "role": "assistant",
                "reasoning_content": "private reasoning is not output",
            }}],
        },
        {
            "id": "stream-response",
            "model": "glm-requested",
            "created": 1786555200,
            "choices": [{"delta": {"content": "NUMPILOT_"}}],
        },
        {
            "id": "stream-response",
            "model": "glm-requested",
            "created": 1786555200,
            "choices": [{"delta": {"content": "EXACT_EDIT_V1"}}],
        },
        {
            "id": "stream-response",
            "model": "glm-requested",
            "created": 1786555200,
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        },
    )
    response_body = (
        "".join(
            "data: %s\n\n" % json.dumps(chunk)
            for chunk in chunks
        )
        + "data: [DONE]\n\n"
    ).encode("utf-8")
    captured = {}

    def fake_urlopen(request, *, timeout, context):
        del timeout, context
        captured["request_data"] = json.loads(request.data.decode("utf-8"))
        return _Response(response_body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    content, receipt = LlmEngine._chat_with_receipt(
        "system",
        "user",
        stream=True,
        thinking_mode="disabled",
        reasoning_effort="minimal",
    )

    assert content == "NUMPILOT_EXACT_EDIT_V1"
    assert captured["request_data"]["stream"] is True
    assert captured["request_data"]["thinking"] == {"type": "disabled"}
    assert captured["request_data"]["reasoning_effort"] == "minimal"
    assert receipt["response_model"] == "glm-requested"
    assert receipt["response_model_identity"] == "exact"
    assert receipt["usage"] == {
        "completion_tokens": 4,
        "prompt_tokens": 10,
    }
    assert receipt["response_body_sha256"] == hashlib.sha256(
        response_body
    ).hexdigest()
    assert receipt["content_sha256"] == hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def test_streaming_completion_stops_at_done_on_keep_alive_connection(
    monkeypatch,
):
    _configure(monkeypatch)
    release_handler = threading.Event()
    event_sent = threading.Event()
    response_body = (
        "data: %s\n\n" % json.dumps({
            "model": "glm-requested",
            "choices": [{"delta": {"content": "ready"}}],
        })
        + "data: [DONE]\n\n"
    ).encode("utf-8")

    class _KeepAliveHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format, *_args):
            return

        def do_POST(self):
            request_size = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(request_size)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.write(
                ("%x\r\n" % len(response_body)).encode("ascii")
                + response_body
                + b"\r\n"
            )
            self.wfile.flush()
            event_sent.set()
            release_handler.wait(5)
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), _KeepAliveHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    monkeypatch.setenv(
        "NUMPILOT_LLM_BASE_URL",
        "http://127.0.0.1:%d/v1" % server.server_port,
    )
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    result = {}

    def invoke():
        try:
            result["value"] = LlmEngine._chat_with_receipt(
                "system", "user", stream=True
            )
        except Exception as exc:  # pragma: no cover - asserted below
            result["error"] = exc

    client_thread = threading.Thread(target=invoke, daemon=True)
    try:
        client_thread.start()
        assert event_sent.wait(2)
        client_thread.join(1)
        completed_before_http_eof = not client_thread.is_alive()
    finally:
        release_handler.set()
        client_thread.join(2)
        server.shutdown()
        server.server_close()
        server_thread.join(2)

    assert completed_before_http_eof
    assert "error" not in result
    content, receipt = result["value"]
    assert content == "ready"
    assert receipt["response_body_sha256"] == hashlib.sha256(
        response_body
    ).hexdigest()


def test_stream_reader_classifies_real_stalled_socket_deadline():
    release_handler = threading.Event()
    headers_sent = threading.Event()

    class _StalledHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format, *_args):
            return

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.flush()
            headers_sent.set()
            release_handler.wait(5)
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), _StalledHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    response = None
    try:
        response = opener.open(
            "http://127.0.0.1:%d/stream" % server.server_port,
            timeout=2,
        )
        assert headers_sent.wait(1)
        started = llm_module.time.monotonic()
        with pytest.raises(LlmTransportError, match="request_timeout"):
            llm_module._read_stream_response_body(
                response,
                deadline=started + 0.1,
            )
        assert llm_module.time.monotonic() - started < 1
    finally:
        if response is not None:
            response.close()
        release_handler.set()
        server.shutdown()
        server.server_close()
        server_thread.join(2)


def test_incomplete_stream_returns_a_receipted_response_error(monkeypatch):
    _configure(monkeypatch)
    response_body = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(response_body),
    )

    with pytest.raises(LlmResponseError, match="incomplete_stream") as caught:
        LlmEngine._chat_with_receipt("system", "user", stream=True)

    assert caught.value.receipt["http_status"] == 200
    assert caught.value.receipt["response_body_sha256"] == hashlib.sha256(
        response_body
    ).hexdigest()


def test_malformed_completed_stream_returns_a_receipted_response_error(
    monkeypatch,
):
    _configure(monkeypatch)
    response_body = b"data: not-json\n\ndata: [DONE]\n\n"
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(response_body),
    )

    with pytest.raises(LlmResponseError, match="invalid_stream_json") as caught:
        LlmEngine._chat_with_receipt("system", "user", stream=True)

    assert caught.value.receipt["response_body_sha256"] == hashlib.sha256(
        response_body
    ).hexdigest()


def test_oversized_stream_fails_during_bounded_incremental_read(monkeypatch):
    _configure(monkeypatch)
    response_body = b"x" * (llm_module._MAX_RESPONSE_BODY_BYTES + 1)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(response_body),
    )

    with pytest.raises(LlmTransportError, match="response_too_large"):
        LlmEngine._chat_with_receipt("system", "user", stream=True)


def test_non_stream_reader_enforces_total_deadline_between_chunks(monkeypatch):
    class _DripResponse:
        def __init__(self):
            self.read_sizes = []

        def read(self, size=-1):
            self.read_sizes.append(size)
            return b"x"

    response = _DripResponse()
    ticks = iter((100.0, 100.1, 100.2))
    monkeypatch.setattr(llm_module.time, "monotonic", lambda: next(ticks))

    with pytest.raises(LlmTransportError, match="request_timeout"):
        llm_module._read_response_body(response, deadline=100.15)

    assert response.read_sizes == [llm_module._MAX_RESPONSE_READ_CHUNK_BYTES]


def test_non_stream_socket_timeout_is_classified_as_request_deadline(
    monkeypatch,
):
    _configure(monkeypatch)

    class _TimedOutResponse(_Response):
        def read(self, size=-1):
            del size
            raise TimeoutError("socket details must stay redacted")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _TimedOutResponse(b""),
    )

    with pytest.raises(LlmTransportError, match="request_timeout") as caught:
        LlmEngine._chat_with_receipt("system", "user")

    assert "socket details" not in str(caught.value)


def test_streaming_timeout_is_a_total_request_deadline(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setenv("NUMPILOT_LLM_TIMEOUT", "10")
    response = _Response(b": heartbeat\n\n")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: response,
    )
    ticks = iter((100.0, 100.1, 110.1))
    monkeypatch.setattr(llm_module.time, "monotonic", lambda: next(ticks))

    with pytest.raises(LlmTransportError, match="request_timeout"):
        LlmEngine._chat_with_receipt("system", "user", stream=True)


def test_streaming_socket_timeout_is_classified_as_request_deadline(
    monkeypatch,
):
    _configure(monkeypatch)

    class _TimedOutResponse(_Response):
        def readline(self, size=-1):
            del size
            raise TimeoutError("socket details must stay redacted")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _TimedOutResponse(b""),
    )

    with pytest.raises(LlmTransportError, match="request_timeout") as caught:
        LlmEngine._chat_with_receipt("system", "user", stream=True)

    assert "socket details" not in str(caught.value)


@pytest.mark.parametrize(
    "transport_error",
    (
        TimeoutError("connect details must stay redacted"),
        urllib.error.URLError(
            TimeoutError("wrapped connect details must stay redacted")
        ),
    ),
)
def test_connection_timeout_is_classified_as_request_deadline(
    monkeypatch, transport_error
):
    _configure(monkeypatch)

    def timed_out_urlopen(*_args, **_kwargs):
        raise transport_error

    monkeypatch.setattr("urllib.request.urlopen", timed_out_urlopen)

    with pytest.raises(LlmTransportError, match="request_timeout") as caught:
        LlmEngine._chat_with_receipt("system", "user", stream=True)

    assert "connect details" not in str(caught.value)


@pytest.mark.parametrize("stream", (None, 1, 0, "true", []))
def test_invalid_per_call_stream_fails_before_transport(monkeypatch, stream):
    _configure(monkeypatch)

    def forbidden_urlopen(*_args, **_kwargs):
        raise AssertionError("invalid per-call stream reached transport")

    monkeypatch.setattr("urllib.request.urlopen", forbidden_urlopen)

    with pytest.raises(LlmConfigError, match="per-call LLM stream"):
        LlmEngine._chat_with_receipt("system", "user", stream=stream)


def test_chat_compatibility_wrapper_still_returns_text(monkeypatch):
    monkeypatch.setattr(
        LlmEngine,
        "_chat_with_receipt",
        staticmethod(lambda _system, _user: ("content", {"receipt": True})),
    )

    assert LlmEngine._chat("system", "user") == "content"


def test_http_error_exposes_only_status_and_never_reads_body(monkeypatch):
    _configure(monkeypatch)

    class _UnreadBody(io.BytesIO):
        read_called = False

        def read(self, *args, **kwargs):
            self.read_called = True
            return super().read(*args, **kwargs)

    body = _UnreadBody(b"sensitive-response-body top-secret-token")
    error = urllib.error.HTTPError(
        "https://private.example/v1/chat/completions",
        429,
        "sensitive-response-body top-secret-token",
        {},
        body,
    )

    def fake_urlopen(_request, *, timeout, context):
        del timeout, context
        raise error

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(LlmHttpError) as caught:
        LlmEngine._chat_with_receipt("system", "user")

    assert str(caught.value) == "remote LLM HTTP status 429"
    assert not body.read_called
    assert "top-secret-token" not in str(caught.value)
    assert "private.example" not in str(caught.value)
    assert "sensitive-response-body" not in str(caught.value)


def test_returned_non_success_status_never_reads_response_body(monkeypatch):
    _configure(monkeypatch)

    class _UnreadResponse(_Response):
        read_called = False

        def read(self, size=-1):
            self.read_called = True
            return super().read(size)

    response = _UnreadResponse(b"sensitive body", status=503)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda _request, *, timeout, context: response,
    )

    with pytest.raises(LlmHttpError, match="HTTP status 503"):
        LlmEngine._chat_with_receipt("system", "user")

    assert not response.read_called


def test_successful_http_with_invalid_json_still_carries_redacted_receipt(
    monkeypatch
):
    _configure(monkeypatch)
    response_body = b"not-json top-secret-token https://private.example/v1"

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda _request, *, timeout, context: _Response(
            response_body,
            headers={"X-Request-Id": "request-top-secret-token"},
        ),
    )

    with pytest.raises(LlmResponseError) as caught:
        LlmEngine._chat_with_receipt("system", "user")

    receipt = caught.value.receipt
    assert str(caught.value) == "remote LLM response is invalid: invalid_json"
    assert receipt["http_status"] == 200
    assert receipt["response_body_sha256"] == hashlib.sha256(
        response_body
    ).hexdigest()
    assert receipt["content_sha256"] is None
    assert receipt["request_id"] == "request-redacted"
    serialized = json.dumps(receipt, sort_keys=True)
    assert "top-secret-token" not in serialized
    assert "https://private.example/v1" not in serialized


def test_receipt_marks_explicitly_disabled_tls_verification(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setenv("NUMPILOT_LLM_INSECURE", "1")
    response_body = json.dumps({
        "choices": [{"message": {"content": "ok"}}],
    }).encode("utf-8")

    def fake_urlopen(_request, *, timeout, context):
        del timeout
        assert context is not None
        return _Response(response_body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    _content, receipt = LlmEngine._chat_with_receipt("system", "user")

    assert receipt["tls_verification"] == "disabled"
