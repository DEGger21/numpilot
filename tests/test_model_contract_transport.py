import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import urllib.request

import pytest

from numpilot.contracts import OffloadPlan
from numpilot.engines.llm import LlmEngine, LlmError
from numpilot.gaps.pipeline import GapCompletionPipeline
from numpilot.gaps.provider import LlmDraftProvider
from numpilot.llm_config import LlmConfigError, load_llm_config, save_llm_config
from numpilot.model_contract import validate_configured_model_receipt


MODEL = "org/transport-model"
PROVIDER = "local-gateway"
TOKEN = "fake-transport-token-secret"
PROXY_VALUE = "proxy-secret-sentinel"
RESPONSE_BODY_SECRET = "response-body-secret-sentinel"

_MODEL_ENV = (
    "NUMPILOT_LLM_BASE_URL",
    "NUMPILOT_LLM_AUTH_TOKEN",
    "NUMPILOT_LLM_MODEL",
    "NUMPILOT_LLM_PROVIDER",
    "NUMPILOT_LLM_PROTOCOL",
    "NUMPILOT_LLM_RESPONSE_MODEL_ALIASES",
    "NUMPILOT_LLM_CONFIG_FILE",
    "NUMPILOT_LLM_MAX_TOKENS",
    "NUMPILOT_LLM_TIMEOUT",
    "NUMPILOT_LLM_INSECURE",
    "NUMPILOT_LLM_REQUIRE_PROXY",
)
_PROXY_ENV = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)

GAP_SOURCE = """\
import numpy as np

def build_inputs():
    return (np.ones(8, dtype=np.float32),)

def forward(x):
    return np.cbrt(x)
"""


def _valid_bundle():
    return json.dumps({
        "summary": "Unverified cbrt registry-invoke draft.",
        "files": [
            {
                "path": "CMakeLists.txt",
                "content": "cmake_minimum_required(VERSION 3.16)\n",
            },
            {
                "path": "build.sh",
                "content": "#!/usr/bin/env bash\nset -euo pipefail\n",
            },
            {"path": "op_host/cbrt_def.cpp", "content": "// host registration\n"},
            {"path": "op_kernel/cbrt.cpp", "content": "// __aicore__ kernel\n"},
            {"path": "op_api/aclnn_cbrt.cpp", "content": "// ACLNN wrapper\n"},
            {"path": "tests/test_cbrt.py", "content": "# NumPy golden test\n"},
            {
                "path": "asnumpy_adapter/operator.json",
                "content": '{"python_name":"cbrt"}\n',
            },
        ],
    })


def _chat_response(content, *, include_model=True, response_id="fake-response"):
    response = {
        "id": response_id,
        "created": 1787702400,
        "choices": [{"message": {"content": content}}],
        "usage": {"total_tokens": 7},
    }
    if include_model:
        response["model"] = MODEL
    return response


class _FakeEndpoint:
    def __init__(self, responses, *, request_id="fake-request"):
        self.responses = list(responses)
        self.request_id = request_id
        self.request_count = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                owner.request_count += 1
                response = owner.responses.pop(0)
                payload = (
                    response
                    if isinstance(response, bytes)
                    else json.dumps(response).encode("utf-8")
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Request-Id", owner.request_id)
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


class _FakeHttpProxy:
    """A local forward-proxy double that terminates the request itself."""

    def __init__(self, response):
        self.response = response
        self.request_count = 0
        self.request_targets = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                owner.request_count += 1
                owner.request_targets.append(self.path)
                payload = json.dumps(owner.response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Request-Id", "fake-proxy-request")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _format, *_args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        host, port = self.server.server_address
        return "http://%s:%d" % (host, port)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def isolated_model_environment(tmp_path, monkeypatch):
    for name in _MODEL_ENV + _PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "config" / "llm.json"
    monkeypatch.setenv("NUMPILOT_LLM_CONFIG_FILE", str(path))
    return path


def _install_direct_urlopen(monkeypatch):
    """Use the local HTTP server without consulting process proxy settings."""

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def direct_urlopen(request, *, timeout, context):
        del context
        return opener.open(request, timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", direct_urlopen)


def _configure(endpoint, path):
    save_llm_config(
        endpoint.base_url,
        TOKEN,
        MODEL,
        provider=PROVIDER,
        path=str(path),
    )
    return load_llm_config(str(path))


def _configure_url(base_url, path):
    save_llm_config(
        base_url,
        TOKEN,
        MODEL,
        provider=PROVIDER,
        path=str(path),
    )
    return load_llm_config(str(path))


def test_required_proxy_routes_request_through_explicit_proxy_snapshot(
    isolated_model_environment, monkeypatch
):
    target = "http://model-origin.invalid/v1"
    with _FakeHttpProxy(_chat_response("proxied-response")) as proxy:
        monkeypatch.setenv("NUMPILOT_LLM_REQUIRE_PROXY", "1")
        monkeypatch.setenv("http_proxy", proxy.url)
        status = _configure_url(target, isolated_model_environment)

        actual, receipt = LlmEngine._chat_with_receipt("system", "user")
        validated = validate_configured_model_receipt(
            receipt,
            actual,
            expected_model=status.model,
            expected_provider=status.provider,
            expected_protocol=status.protocol,
            response_model_aliases=status.response_model_aliases,
        )

    assert actual == "proxied-response"
    assert validated["http_status"] == 200
    assert proxy.request_count == 1
    assert proxy.request_targets == [
        target + "/chat/completions"
    ]


def test_required_proxy_fails_before_transport_when_matching_proxy_is_missing(
    isolated_model_environment, monkeypatch
):
    _configure_url(
        "https://model-origin.invalid/v1",
        isolated_model_environment,
    )
    monkeypatch.setenv("NUMPILOT_LLM_REQUIRE_PROXY", "1")
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})

    def forbidden_transport(*_args, **_kwargs):
        raise AssertionError("network transport must not start")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_transport)
    monkeypatch.setattr(urllib.request, "build_opener", forbidden_transport)

    with pytest.raises(LlmConfigError, match="no matching HTTP.S. proxy"):
        LlmEngine._chat_with_receipt("system", "user")


def test_required_proxy_fails_before_transport_when_endpoint_is_bypassed(
    isolated_model_environment, monkeypatch
):
    _configure_url(
        "https://model-origin.invalid/v1",
        isolated_model_environment,
    )
    monkeypatch.setenv("NUMPILOT_LLM_REQUIRE_PROXY", "1")
    monkeypatch.setattr(
        urllib.request,
        "getproxies",
        lambda: {"https": "http://proxy-secret.invalid:8080"},
    )
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda _host: True)

    def forbidden_transport(*_args, **_kwargs):
        raise AssertionError("network transport must not start")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_transport)
    monkeypatch.setattr(urllib.request, "build_opener", forbidden_transport)

    with pytest.raises(LlmConfigError, match="proxy bypass policy") as caught:
        LlmEngine._chat_with_receipt("system", "user")
    assert "proxy-secret" not in str(caught.value)


def test_public_engine_rejects_response_without_model(
    isolated_model_environment, monkeypatch
):
    migrated = """\
import asnumpy as ap

def forward(x):
    return ap.exp(x)
"""
    response = _chat_response(
        "```python\n%s```" % migrated,
        include_model=False,
    )
    with _FakeEndpoint([response]) as endpoint:
        _install_direct_urlopen(monkeypatch)
        _configure(endpoint, isolated_model_environment)
        engine = LlmEngine()

        with pytest.raises(LlmError, match="response_model is missing"):
            engine.rewrite(
                "import numpy as np\n\ndef forward(x):\n    return np.exp(x)\n",
                OffloadPlan(placement={"exp": "NPU"}, threshold_elements=1),
            )

    assert endpoint.request_count == 1
    assert engine.last_receipt == {}
    assert engine.last_identity == {}


@pytest.mark.parametrize("first_failure", ("invalid_json", "missing_model"))
def test_failed_response_receipt_survives_retry_publish_and_offline_reload(
    isolated_model_environment, monkeypatch, tmp_path, first_failure
):
    bundle = _valid_bundle()
    first_response = (
        b"not-json " + RESPONSE_BODY_SECRET.encode("ascii")
        if first_failure == "invalid_json"
        else _chat_response(bundle, include_model=False)
    )
    with _FakeEndpoint([
        first_response,
        _chat_response(bundle),
    ]) as endpoint:
        _install_direct_urlopen(monkeypatch)
        status = _configure(endpoint, isolated_model_environment)
        source = tmp_path / (first_failure + "-cbrt.py")
        source.write_text(GAP_SOURCE, encoding="utf-8")
        out = tmp_path / (first_failure + "-out")
        provider = LlmDraftProvider(
            max_attempts=2,
            configuration_status=status,
        )

        generated = GapCompletionPipeline(provider=provider).run(
            str(source),
            str(out),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )

        evidence = generated.drafts[0].generation
        assert evidence["attempt_count"] == 2
        assert evidence["transport_attempt_count"] == 2
        assert evidence["successful_http_response_count"] == 2
        assert evidence["content_response_count"] == (
            1 if first_failure == "invalid_json" else 2
        )
        assert len(evidence["remote_receipts"]) == 2
        failed_receipt, successful_receipt = evidence["remote_receipts"]
        assert failed_receipt["response_model"] is None
        assert failed_receipt["response_model_identity"] == "missing"
        assert failed_receipt["content_sha256"] == (
            None
            if first_failure == "invalid_json"
            else hashlib.sha256(bundle.encode("utf-8")).hexdigest()
        )
        assert successful_receipt["response_model"] == MODEL
        assert successful_receipt["response_model_identity"] == "exact"

        for name in _MODEL_ENV[:-1]:
            monkeypatch.delenv(name, raising=False)
        offline_provider = LlmDraftProvider(
            configuration_loader=lambda: (_ for _ in ()).throw(
                AssertionError("offline reload must not load model credentials")
            )
        )
        reloaded = GapCompletionPipeline(provider=offline_provider).run(
            str(source),
            str(out),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )

    assert endpoint.request_count == 2
    assert offline_provider.action_for(reloaded.drafts[0]) == "reused_existing"
    assert reloaded.drafts[0].artifact_id == generated.drafts[0].artifact_id


def test_transport_receipt_and_logs_exclude_proxy_endpoint_token_and_body(
    isolated_model_environment, monkeypatch, capsys, caplog
):
    content = RESPONSE_BODY_SECRET
    with _FakeEndpoint(
        [_chat_response(
            content,
            response_id="response-%s-%s" % (TOKEN, PROXY_VALUE),
        )],
        request_id="request-%s-%s" % (PROXY_VALUE, TOKEN),
    ) as endpoint:
        _install_direct_urlopen(monkeypatch)
        status = _configure(endpoint, isolated_model_environment)
        for name in _PROXY_ENV:
            monkeypatch.setenv(name, PROXY_VALUE)

        actual, receipt = LlmEngine._chat_with_receipt("system", "user")
        validated = validate_configured_model_receipt(
            receipt,
            actual,
            expected_model=status.model,
            expected_provider=status.provider,
            expected_protocol=status.protocol,
            response_model_aliases=status.response_model_aliases,
        )

    assert endpoint.request_count == 1
    assert validated["content_sha256"] == hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()
    serialized = json.dumps(validated, ensure_ascii=False, sort_keys=True)
    captured = capsys.readouterr()
    logs = caplog.text + captured.out + captured.err
    for secret in (TOKEN, endpoint.base_url, PROXY_VALUE, RESPONSE_BODY_SECRET):
        assert secret not in serialized
        assert secret not in logs
