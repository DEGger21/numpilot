import hashlib
from pathlib import Path

import pytest

from numpilot.engines.llm import (
    LlmHttpError,
    LlmResponseError,
    LlmTransportError,
)
from numpilot.llm_config import (
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    LlmConfigStatus,
    model_identity_sha256,
)
from numpilot.model_gateway import (
    ModelGateway,
    ModelGatewayError,
    ModelRequest,
)


CONTENT = '{"ok":true}'
MODEL = "org/model"
PROVIDER = "gateway"


def _status(*, configured=True):
    return LlmConfigStatus(
        path=Path("/tmp/llm.json"),
        configured=configured,
        missing=() if configured else ("NUMPILOT_LLM_MODEL",),
        source="test",
        model=MODEL if configured else None,
        provider=PROVIDER if configured else None,
        token_present=configured,
        protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        response_model_aliases=(),
    )


def _receipt(content=CONTENT, *, response_model=MODEL):
    identity = (
        "exact" if response_model == MODEL else
        "missing" if response_model is None else "mismatch"
    )
    return {
        "schema_version": 2,
        "utc_start": "2026-10-31T00:00:00Z",
        "utc_end": "2026-10-31T00:00:01Z",
        "http_status": 200,
        "requested_model": MODEL,
        "response_model": response_model,
        "response_id": "fake-response",
        "response_created": 1,
        "request_body_sha256": "a" * 64,
        "response_body_sha256": "b" * 64,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "request_id": "fake-request",
        "usage": {"total_tokens": 1},
        "tls_verification": "enabled",
        "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        "provider": PROVIDER,
        "configured_model": MODEL,
        "model_identity_sha256": model_identity_sha256(
            protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            provider=PROVIDER,
            configured_model=MODEL,
            requested_model=MODEL,
            response_model=response_model,
            response_model_aliases=(),
            response_model_identity=identity,
        ),
        "response_model_aliases": [],
        "response_model_identity": identity,
    }


def test_gateway_validates_once_and_forwards_bounded_options():
    calls = []
    loads = []

    def loader():
        loads.append(True)
        return _status()

    def transport(system, user, **options):
        calls.append((system, user, options))
        return CONTENT, _receipt()

    gateway = ModelGateway(configuration_loader=loader, transport=transport)
    request = ModelRequest(
        task="script_migration",
        role="code_migrator",
        system="system",
        user="user",
        max_tokens=42,
        thinking_mode="disabled",
        reasoning_effort="high",
        stream=True,
        source_sha256="c" * 64,
    )

    result = gateway.complete(request)

    assert len(loads) == 1
    assert calls == [
        ("system", "user", {
            "max_tokens": 42,
            "thinking_mode": "disabled",
            "reasoning_effort": "high",
            "stream": True,
        })
    ]
    assert result.content == CONTENT
    assert result.receipt["response_model_identity"] == "exact"
    assert result.actual_remote_call is False
    assert result.response_sha256 == hashlib.sha256(CONTENT.encode()).hexdigest()
    assert result.content_sha256 == result.response_sha256
    assert result.response_body_sha256 == "b" * 64
    assert len(result.request_sha256) == 64


def test_gateway_keeps_mismatch_as_valid_attempt_evidence():
    def transport(_system, _user, **_options):
        return CONTENT, _receipt(response_model="gateway/alias")

    result = ModelGateway(
        configuration_loader=lambda: _status(),
        transport=transport,
    ).complete(ModelRequest("gap_generation", "system", "user"))

    assert result.receipt["response_model_identity"] == "mismatch"
    assert result.identity.model == MODEL


def test_invalid_task_fails_before_configuration_or_transport():
    calls = []
    with pytest.raises(ValueError, match="unsupported model task"):
        ModelRequest("unknown_task", "system", "user")
    with pytest.raises(ValueError):
        ModelRequest("script_migration", "system", "user", max_tokens=0)

    def loader():
        calls.append("loader")
        return _status()

    gateway = ModelGateway(configuration_loader=loader, transport=lambda *_a, **_k: None)
    assert calls == []
    assert gateway is not None


def test_missing_configuration_never_calls_transport():
    calls = []
    gateway = ModelGateway(
        configuration_loader=lambda: _status(configured=False),
        transport=lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(ModelGatewayError) as exc_info:
        gateway.complete(ModelRequest("readiness_gate", "system", "user"))

    assert exc_info.value.category == "config"
    assert calls == []


def test_transport_failures_are_safe():
    secret = "top-secret-token"
    endpoint = "https://private.example/v1"

    def transport(_system, _user, **_options):
        raise RuntimeError(secret + " " + endpoint)

    gateway = ModelGateway(
        configuration_loader=lambda: _status(),
        transport=transport,
    )
    with pytest.raises(ModelGatewayError) as exc_info:
        gateway.complete(ModelRequest("gap_repair", "system", "user"))

    assert exc_info.value.category == "transport"
    assert secret not in str(exc_info.value)
    assert endpoint not in str(exc_info.value)


@pytest.mark.parametrize(
    ("error", "category", "status_code"),
    (
        (LlmHttpError(503), "http", 503),
        (LlmTransportError("timeout"), "transport", None),
        (LlmResponseError("invalid", _receipt()), "response", None),
    ),
)
def test_known_transport_errors_are_classified_without_raw_details(
    error, category, status_code
):
    gateway = ModelGateway(
        configuration_loader=lambda: _status(),
        transport=lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(ModelGatewayError) as exc_info:
        gateway.complete(ModelRequest("connectivity_check", "system", "user"))

    assert exc_info.value.category == category
    assert exc_info.value.status_code == status_code
    assert "top-secret-token" not in str(exc_info.value)


def test_invalid_receipt_is_rejected_at_gateway_boundary():
    invalid = _receipt()
    invalid["extra"] = "must fail"
    gateway = ModelGateway(
        configuration_loader=lambda: _status(),
        transport=lambda *_args, **_kwargs: (CONTENT, invalid),
    )

    with pytest.raises(ModelGatewayError) as exc_info:
        gateway.complete(ModelRequest("readiness_gate", "system", "user"))

    assert exc_info.value.category == "contract"
