import hashlib
import io
import json
import urllib.error

import pytest

from numpilot.engines.llm import LlmEngine, LlmHttpError, LlmResponseError


class _Response:
    def __init__(self, payload, *, status=200, headers=None):
        self._payload = payload
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def read(self, size=-1):
        if size is None or size < 0:
            return self._payload
        return self._payload[:size]

    def getcode(self):
        return self.status


def _configure(monkeypatch):
    monkeypatch.setenv("NUMPILOT_LLM_BASE_URL", "https://private.example/v1")
    monkeypatch.setenv("NUMPILOT_LLM_AUTH_TOKEN", "top-secret-token")
    monkeypatch.setenv("NUMPILOT_LLM_MODEL", "glm-requested")
    monkeypatch.delenv("NUMPILOT_LLM_INSECURE", raising=False)


def test_remote_chat_returns_content_and_redacted_audit_receipt(monkeypatch):
    _configure(monkeypatch)
    content = '{"summary":"ok","files":[]}'
    response_body = json.dumps({
        "id": "response-top-secret-token",
        "model": "glm-response-https://private.example/v1",
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
    assert receipt["schema_version"] == 1
    assert receipt["utc_start"].endswith("Z")
    assert receipt["utc_end"].endswith("Z")
    assert receipt["http_status"] == 200
    assert receipt["requested_model"] == "glm-requested"
    assert receipt["response_model"] == "glm-response-redacted"
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
