import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread


ROOT = Path(__file__).resolve().parents[1]
MODEL = "org/model"
PROVIDER = "generic-gateway"
PROTOCOL = "openai-chat-completions"
TOKEN = "loopback-only-test-token"

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


class _FakeOpenAIEndpoint:
    def __init__(self):
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
                payload = json.dumps({
                    "id": "direct-lane-response",
                    "created": 1787702400,
                    "model": request["model"],
                    "choices": [{
                        "message": {
                            "content": "```python\n" + MIGRATED + "```",
                        },
                    }],
                    "usage": {"total_tokens": 17},
                }).encode("utf-8")
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


_DIRECT_CLI_CHILD = r"""
import sys
import types
from importlib.abc import MetaPathFinder

import numpy as np


class _RejectCannbotImport(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "numpilot.cannbot_skills" or fullname.startswith(
            "numpilot.cannbot_skills."
        ):
            raise ModuleNotFoundError(
                "CANNBot is intentionally unavailable to the direct lane",
                name=fullname,
            )
        return None


sys.meta_path.insert(0, _RejectCannbotImport())

from numpilot import __main__ as cli
from numpilot.capabilities import (
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCapability,
)


class _DirectBackend:
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
            backend_version="isolated-direct-test",
            audit_id="isolated-direct-test-v1",
            environment={},
            operators={
                name: OperatorCapability(
                    name=name,
                    available=True,
                    provenance=ExecutionProvenance.NATIVE_NPU,
                    evidence="isolated direct-lane capability",
                )
                for name in required_ops or ()
            },
        )


cli._qualified_runtime_inputs = lambda _args, _name: (_DirectBackend(), None)
cli._preflight = lambda _backend, _skip: None
raise SystemExit(cli.main([
    "migrate",
    sys.argv[1],
    "--backend", "npu",
    "--out", sys.argv[2],
]))
"""


def test_direct_script_migration_needs_no_cannbot_module_or_checkout(tmp_path):
    source = tmp_path / "source.py"
    source.write_text(SOURCE, encoding="utf-8")
    out = tmp_path / "out"
    missing_cannbot = tmp_path / "missing-cannbot-checkout"

    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("NUMPILOT_LLM_") or name.lower() in {
            "http_proxy",
            "https_proxy",
            "all_proxy",
        }:
            environment.pop(name, None)
    environment.update({
        "HOME": str(tmp_path / "empty-home"),
        "XDG_CACHE_HOME": str(tmp_path / "empty-cache"),
        "NUMPILOT_CANNBOT_CACHE": str(missing_cannbot),
        "NUMPILOT_CONFIG_HOME": str(tmp_path / "empty-config"),
        "NUMPILOT_STATE_HOME": str(tmp_path / "state"),
        "NUMPILOT_LLM_AUTH_TOKEN": TOKEN,
        "NUMPILOT_LLM_MODEL": MODEL,
        "NUMPILOT_LLM_PROVIDER": PROVIDER,
        "NUMPILOT_LLM_PROTOCOL": PROTOCOL,
        "NUMPILOT_LLM_RESPONSE_MODEL_ALIASES": "[]",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "PYTHONDONTWRITEBYTECODE": "1",
    })

    with _FakeOpenAIEndpoint() as endpoint:
        environment["NUMPILOT_LLM_BASE_URL"] = endpoint.base_url
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(_DIRECT_CLI_CHILD),
                str(source),
                str(out),
            ],
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert len(endpoint.requests) == 1
    request = endpoint.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["authorization"] == "Bearer " + TOKEN
    assert request["json"]["model"] == MODEL
    assert (out / "migrated.py").read_text(encoding="utf-8") == MIGRATED

    report = json.loads(
        (out / "script-migration-report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "migrated_strict_npu"
    assert report["generated_by"] == "configured_model"
    assert report["provider"] == PROVIDER
    assert report["protocol"] == PROTOCOL
    assert report["configured_model"] == MODEL
    assert report["requested_model"] == MODEL
    assert report["response_model"] == MODEL
    assert report["llm_receipt"]["configured_model"] == MODEL
    assert report["llm_receipt"]["response_model_identity"] == "exact"
    assert not missing_cannbot.exists()
    assert TOKEN not in completed.stdout + completed.stderr
    assert MIGRATED not in completed.stdout + completed.stderr
