import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import types

import numpy as np
import pytest

from numpilot import __main__ as cli
from numpilot.capabilities import (
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCapability,
)
from numpilot.engines.llm import LlmEngine
from numpilot.gaps.provider import (
    DraftGenerationError,
    validate_configured_model_receipt,
)
from numpilot.llm_config import (
    LlmConfigError,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    load_llm_config,
    model_identity_sha256,
    save_llm_config,
)
from numpilot.script_migration import (
    ScriptMigrationError,
    classify_script,
    generate_script_with_configured_model,
    scan_numpy_script,
)


SOURCE = """\
import numpy as np
x = np.full((2,), 1.0, dtype=np.float32)
y = np.exp(x)
print(y)
"""

MIGRATED = """\
import asnumpy as ap
x = ap.full((2,), 1.0, dtype=ap.float32)
y = ap.exp(x)
print(y)
"""

MODEL_ENV = (
    "NUMPILOT_LLM_BASE_URL",
    "NUMPILOT_LLM_AUTH_TOKEN",
    "NUMPILOT_LLM_MODEL",
    "NUMPILOT_LLM_PROVIDER",
    "NUMPILOT_LLM_PROTOCOL",
    "NUMPILOT_LLM_RESPONSE_MODEL_ALIASES",
    "NUMPILOT_LLM_TIMEOUT",
    "NUMPILOT_LLM_CONFIG_FILE",
)


class _Backend:
    name = "npu"
    module = types.SimpleNamespace(
        full=lambda *args, **kwargs: None,
        exp=lambda *args, **kwargs: None,
        float32=np.float32,
    )

    def get_module(self):
        return self.module

    def capability_manifest(self, required_ops=None):
        return CapabilityManifest(
            backend="npu",
            backend_version="fake-runtime",
            audit_id="fake-audit-v1",
            environment={},
            operators={
                name: OperatorCapability(
                    name=name,
                    available=True,
                    provenance=ExecutionProvenance.NATIVE_NPU,
                    evidence="local fake capability",
                )
                for name in required_ops or ()
            },
        )


class _FakeEndpoint:
    def __init__(self, response_model=None):
        self.response_model = response_model
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                request = json.loads(self.rfile.read(length))
                owner.requests.append({
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "json": request,
                })
                prompt = request["messages"][-1]["content"]
                content = (
                    "NUMPILOT_LLM_READY."
                    if "NUMPILOT_LLM_READY" in prompt
                    else "```python\n" + MIGRATED + "```"
                )
                payload = json.dumps({
                    "id": "fake-response",
                    "created": 1787659200,
                    "model": (
                        owner.response_model
                        if owner.response_model is not None
                        else request["model"]
                    ),
                    "choices": [{"message": {"content": content}}],
                    "usage": {"total_tokens": 7},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _format, *_args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        host, port = self.server.server_address
        return "http://%s:%d/v1" % (host, port)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def isolated_model_config(tmp_path, monkeypatch):
    for key in MODEL_ENV:
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "config" / "llm.json"
    monkeypatch.setenv("NUMPILOT_LLM_CONFIG_FILE", str(path))
    return path


def _reload_saved_config(monkeypatch):
    for key in MODEL_ENV[:-1]:
        monkeypatch.delenv(key, raising=False)
    return load_llm_config()


@pytest.mark.parametrize(
    ("model", "provider"),
    (
        ("glm-5.3", "glm"),
        ("gpt-5", "openai"),
        ("deepseek-chat", "deepseek"),
        ("org/model", "gateway-audit-label"),
    ),
)
def test_configure_check_probe_and_generate_share_generic_contract(
    isolated_model_config, monkeypatch, capsys, model, provider
):
    token = "fake-loopback-token"
    with _FakeEndpoint() as endpoint:
        path = save_llm_config(
            endpoint.base_url,
            token,
            model,
            provider=provider,
            protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        )
        assert path == isolated_model_config
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["schema_version"] == 2
        assert saved["protocol"] == OPENAI_CHAT_COMPLETIONS_PROTOCOL
        assert saved["response_model_aliases"] == []

        status = _reload_saved_config(monkeypatch)
        assert status.model == model
        assert status.provider == provider
        assert status.protocol == OPENAI_CHAT_COMPLETIONS_PROTOCOL
        assert status.response_model_aliases == ()

        assert cli.main(["configure-llm", "--check"]) == 0
        assert cli.main(["configure-llm", "--probe"]) == 0

        analysis = classify_script(scan_numpy_script(SOURCE), _Backend())
        artifact = generate_script_with_configured_model(
            analysis,
            expected_model=status.model,
            expected_provider=status.provider,
            expected_protocol=status.protocol,
            response_model_aliases=status.response_model_aliases,
        )

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert token not in output
    assert MIGRATED not in output
    assert artifact.model == model
    assert artifact.response_model == model
    assert artifact.provider == provider
    assert artifact.protocol == OPENAI_CHAT_COMPLETIONS_PROTOCOL
    assert artifact.receipt["schema_version"] == 2
    assert artifact.receipt["configured_model"] == model
    assert [item["json"]["model"] for item in endpoint.requests] == [model, model]
    assert all(item["path"] == "/v1/chat/completions" for item in endpoint.requests)
    assert all(item["authorization"] == "Bearer " + token for item in endpoint.requests)


def test_response_model_alias_is_rejected_by_default_and_accepted_only_when_explicit(
    isolated_model_config, monkeypatch
):
    configured = "org/model"
    alias = "gateway/model-alias"
    with _FakeEndpoint(response_model=alias) as endpoint:
        save_llm_config(
            endpoint.base_url,
            "fake-token",
            configured,
            provider="gateway",
        )
        status = _reload_saved_config(monkeypatch)
        assert cli.main(["configure-llm", "--probe"]) == 4

        save_llm_config(
            endpoint.base_url,
            "fake-token",
            configured,
            provider="gateway",
            response_model_aliases=(alias,),
        )
        status = _reload_saved_config(monkeypatch)
        assert status.response_model_aliases == (alias,)
        assert cli.main(["configure-llm", "--probe"]) == 0

        analysis = classify_script(scan_numpy_script(SOURCE), _Backend())
        artifact = generate_script_with_configured_model(
            analysis,
            expected_model=status.model,
            expected_provider=status.provider,
            expected_protocol=status.protocol,
            response_model_aliases=status.response_model_aliases,
        )
        assert artifact.model == configured
        assert artifact.response_model == alias


def test_response_model_with_boundary_whitespace_fails_probe_and_generation(
    isolated_model_config, monkeypatch
):
    configured = "org/model"
    with _FakeEndpoint(response_model=" org/model ") as endpoint:
        save_llm_config(
            endpoint.base_url,
            "fake-token",
            configured,
            provider="gateway",
        )
        status = _reload_saved_config(monkeypatch)
        assert cli.main(["configure-llm", "--probe"]) == 4

        analysis = classify_script(scan_numpy_script(SOURCE), _Backend())
        with pytest.raises(ScriptMigrationError, match="receipt failed validation"):
            generate_script_with_configured_model(
                analysis,
                expected_model=status.model,
                expected_provider=status.provider,
                expected_protocol=status.protocol,
                response_model_aliases=status.response_model_aliases,
            )

    assert len(endpoint.requests) == 2


def test_timeout_profile_is_shared_by_check_probe_and_transport(
    isolated_model_config, monkeypatch
):
    with _FakeEndpoint() as endpoint:
        save_llm_config(
            endpoint.base_url,
            "fake-token",
            "org/model",
            provider="gateway",
        )
        monkeypatch.setenv("NUMPILOT_LLM_TIMEOUT", "1")
        for key in MODEL_ENV[:-2]:
            if key != "NUMPILOT_LLM_TIMEOUT":
                monkeypatch.delenv(key, raising=False)

        assert cli.main(["configure-llm", "--check"]) == 1
        assert cli.main(["configure-llm", "--probe"]) == 1
        with pytest.raises(LlmConfigError, match="10～3600"):
            LlmEngine._chat_with_receipt("system", "user")

    assert endpoint.requests == []


def _schema2_receipt(content, *, configured="org/model", response=None):
    response = configured if response is None else response
    identity = {
        "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        "provider": "gateway",
        "configured_model": configured,
        "requested_model": configured,
        "response_model": response,
    }
    return {
        "schema_version": 2,
        "protocol": identity["protocol"],
        "provider": identity["provider"],
        "configured_model": identity["configured_model"],
        "requested_model": identity["requested_model"],
        "response_model": identity["response_model"],
        "response_model_aliases": [],
        "response_model_identity": (
            "exact" if response == configured else "mismatch"
        ),
        "model_identity_sha256": hashlib.sha256(
            json.dumps({
                **identity,
                "response_model_aliases": [],
                "response_model_identity": (
                    "exact" if response == configured else "mismatch"
                ),
            }, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "utc_start": "2026-08-26T00:00:00Z",
        "utc_end": "2026-08-26T00:00:01Z",
        "http_status": 200,
        "response_id": "fake-response",
        "response_created": 1787702400,
        "request_body_sha256": "1" * 64,
        "response_body_sha256": "2" * 64,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "request_id": "fake-request",
        "usage": {"total_tokens": 7},
        "tls_verification": "enabled",
    }


def test_generic_receipt_identity_and_content_tampering_fail_closed():
    content = "private response"
    receipt = _schema2_receipt(content)
    validated = validate_configured_model_receipt(
        receipt,
        content,
        expected_model="org/model",
        expected_provider="gateway",
        expected_protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    )
    assert validated["response_model"] == "org/model"

    for field, value in (
        ("configured_model", "other/model"),
        ("requested_model", "other/model"),
        ("response_model", "other/model"),
        ("model_identity_sha256", "0" * 64),
        ("response_model_identity", "explicit_alias"),
        ("content_sha256", "0" * 64),
        ("protocol", "other-protocol"),
        ("provider", "other-label"),
    ):
        tampered = dict(receipt)
        tampered[field] = value
        with pytest.raises(DraftGenerationError):
            validate_configured_model_receipt(
                tampered,
                content,
                expected_model="org/model",
                expected_provider="gateway",
                expected_protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            )


def test_legacy_glm_receipt_is_read_only_compatible_but_not_generic_schema():
    content = "legacy response"
    receipt = {
        "schema_version": 1,
        "utc_start": "2026-08-25T12:00:00Z",
        "utc_end": "2026-08-25T12:00:01Z",
        "http_status": 200,
        "requested_model": "glm-5.3",
        "response_model": "glm-5.3",
        "response_id": "legacy-response",
        "response_created": 1787659200,
        "request_body_sha256": "1" * 64,
        "response_body_sha256": "2" * 64,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "request_id": "legacy-request",
        "usage": {},
        "tls_verification": "enabled",
    }
    assert validate_configured_model_receipt(
        receipt,
        content,
        expected_model="glm-5.3",
        expected_provider="glm",
        expected_protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    )["schema_version"] == 1

    non_glm = dict(receipt)
    non_glm["requested_model"] = non_glm["response_model"] = "gpt-5"
    with pytest.raises(DraftGenerationError, match="legacy GLM"):
        validate_configured_model_receipt(
            non_glm,
            content,
            expected_model="gpt-5",
            expected_provider="openai",
            expected_protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        )


def test_missing_response_model_is_not_silently_accepted():
    receipt = _schema2_receipt("content")
    receipt["response_model"] = None
    with pytest.raises(DraftGenerationError):
        validate_configured_model_receipt(
            receipt,
            "content",
            expected_model="org/model",
            expected_provider="gateway",
            expected_protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        )


def test_failed_schema2_receipt_with_no_content_can_be_reloaded_for_diagnostics():
    receipt = _schema2_receipt("unavailable")
    receipt.update({
        "response_model": None,
        "response_model_identity": "missing",
        "content_sha256": None,
    })
    receipt["model_identity_sha256"] = model_identity_sha256(
        protocol=receipt["protocol"],
        provider=receipt["provider"],
        configured_model=receipt["configured_model"],
        requested_model=receipt["requested_model"],
        response_model=None,
        response_model_aliases=(),
        response_model_identity="missing",
    )

    explicit = validate_configured_model_receipt(
        receipt,
        None,
        expected_model="org/model",
        expected_provider="gateway",
        expected_protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    )
    offline = validate_configured_model_receipt(
        receipt,
        expected_model="org/model",
        expected_provider="gateway",
        expected_protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    )

    assert explicit == offline == receipt


def test_llm_engine_propagates_validated_receipt_and_exact_identity(
    isolated_model_config, monkeypatch
):
    with _FakeEndpoint() as endpoint:
        save_llm_config(
            endpoint.base_url,
            "fake-token",
            "org/model",
            provider="gateway",
        )
        status = _reload_saved_config(monkeypatch)
        assert status.configured
        plan = types.SimpleNamespace(placement={"exp": "NPU"})
        engine = LlmEngine()
        migrated = engine.rewrite(SOURCE, plan)

    assert migrated == MIGRATED.rstrip()
    assert engine.last_receipt["configured_model"] == "org/model"
    assert engine.last_receipt["response_model"] == "org/model"
    assert engine.last_identity["response_model_identity"] == "exact"
    assert engine.last_identity["artifact_sha256"] == hashlib.sha256(
        migrated.encode("utf-8")
    ).hexdigest()


def test_unsupported_protocol_fails_before_remote_generation(
    isolated_model_config, monkeypatch
):
    with pytest.raises(Exception, match="protocol"):
        save_llm_config(
            "https://gateway.example/v1",
            "fake-token",
            "org/model",
            provider="gateway",
            protocol="responses-api",
        )

    assert not isolated_model_config.exists()
    assert os.getenv("NUMPILOT_LLM_AUTH_TOKEN") is None
