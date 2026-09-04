# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
#
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
#
# Authors (accounts):
# - Su Tonghua <@sutonghua>
# - Li Xing <@li-xingHIT>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import io
from pathlib import Path
import re

import pytest

from numpilot.llm_check import (
    LLM_CHECK_MARKER,
    LLM_CHECK_MAX_TOKENS,
    LlmConnectivityCheckError,
    render_llm_connectivity_error,
    render_llm_connectivity_result,
    run_llm_connectivity_check,
    run_llm_check_command,
)
from numpilot.llm_config import (
    LlmConfigError,
    LlmConfigStatus,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    model_identity_sha256,
)


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class _TtyBuffer(io.StringIO):
    def isatty(self):
        return True


def _configured_status():
    return LlmConfigStatus(
        path=Path("/private/llm.json"),
        configured=True,
        missing=(),
        source="user-config",
        model="org/model",
        provider="gateway",
        token_present=True,
        protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        response_model_aliases=(),
    )


def _proxy(*, policy="optional", route="direct"):
    return {
        "policy": policy,
        "scheme": "https",
        "matched": route in {"environment", "enforced"},
        "bypassed": False,
        "route": route,
    }


def _receipt(content, *, response_model="org/model"):
    state = "exact" if response_model == "org/model" else "mismatch"
    return {
        "schema_version": 2,
        "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        "provider": "gateway",
        "configured_model": "org/model",
        "requested_model": "org/model",
        "response_model": response_model,
        "response_model_aliases": [],
        "response_model_identity": state,
        "model_identity_sha256": model_identity_sha256(
            protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            provider="gateway",
            configured_model="org/model",
            requested_model="org/model",
            response_model=response_model,
            response_model_aliases=(),
            response_model_identity=state,
        ),
        "utc_start": "2026-09-04T00:00:00Z",
        "utc_end": "2026-09-04T00:00:01Z",
        "http_status": 200,
        "response_id": "check-response",
        "response_created": 1788480000,
        "request_body_sha256": "1" * 64,
        "response_body_sha256": "2" * 64,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "request_id": "check-request",
        "usage": {"total_tokens": 9},
        "tls_verification": "enabled",
    }


def test_connectivity_check_uses_bounded_prompt_and_renders_no_response_secrets(
    monkeypatch,
):
    import numpilot.llm_check as check

    secret = "never-print-this-token"
    endpoint = "https://private-model.example/v1"
    proxy_url = "http://proxy-user:proxy-pass@127.0.0.1:7897"
    content = "%s %s %s %s" % (LLM_CHECK_MARKER, secret, endpoint, proxy_url)
    call = {}

    monkeypatch.setattr(check, "load_llm_config", lambda _path=None: _configured_status())
    monkeypatch.setattr(
        check,
        "llm_proxy_route_status",
        lambda: _proxy(route="environment"),
    )

    def chat(system, user, **kwargs):
        call.update(system=system, user=user, kwargs=kwargs)
        return content, _receipt(content)

    result = run_llm_connectivity_check(chat=chat)
    rendered = render_llm_connectivity_result(result)
    serialized = repr(result.to_safe_dict())

    assert call["user"] == "Reply with NUMPILOT_LLM_READY."
    assert call["kwargs"] == {"max_tokens": LLM_CHECK_MAX_TOKENS}
    assert result.http_status == 200
    assert result.proxy_route == "environment"
    assert "actual_remote_call=yes" in rendered
    assert "marker=verified" in rendered
    for sensitive in (secret, endpoint, proxy_url, content):
        assert sensitive not in rendered
        assert sensitive not in serialized


def test_connectivity_check_rejects_missing_marker_after_valid_remote_response(
    monkeypatch,
):
    import numpilot.llm_check as check

    content = "hello"
    monkeypatch.setattr(check, "load_llm_config", lambda _path=None: _configured_status())
    monkeypatch.setattr(check, "llm_proxy_route_status", lambda: _proxy())

    with pytest.raises(LlmConnectivityCheckError) as caught:
        run_llm_connectivity_check(
            chat=lambda *_args, **_kwargs: (content, _receipt(content))
        )

    assert caught.value.category == "response_invalid"
    assert caught.value.remote_attempted is True
    assert caught.value.actual_remote_call is True
    assert content not in render_llm_connectivity_error(caught.value)


def test_connectivity_check_blocks_required_proxy_before_transport(monkeypatch):
    import numpilot.llm_check as check

    calls = []
    monkeypatch.setattr(check, "load_llm_config", lambda _path=None: _configured_status())
    monkeypatch.setattr(
        check,
        "llm_proxy_route_status",
        lambda: _proxy(policy="required", route="blocked"),
    )

    with pytest.raises(LlmConnectivityCheckError) as caught:
        run_llm_connectivity_check(
            chat=lambda *_args, **_kwargs: calls.append(True)
        )

    assert caught.value.category == "proxy_blocked"
    assert caught.value.remote_attempted is False
    assert calls == []


def test_connectivity_check_never_forwards_config_or_unexpected_error_text(
    monkeypatch,
):
    import numpilot.llm_check as check

    secret_text = "token=private https://private-endpoint.invalid/v1"
    monkeypatch.setattr(
        check,
        "load_llm_config",
        lambda _path=None: (_ for _ in ()).throw(LlmConfigError(secret_text)),
    )
    with pytest.raises(LlmConnectivityCheckError) as config_error:
        run_llm_connectivity_check()
    assert secret_text not in str(config_error.value)
    assert secret_text not in repr(config_error.value.to_safe_dict())

    monkeypatch.setattr(check, "load_llm_config", lambda _path=None: _configured_status())
    monkeypatch.setattr(check, "llm_proxy_route_status", lambda: _proxy())
    with pytest.raises(LlmConnectivityCheckError) as unexpected_error:
        run_llm_connectivity_check(
            chat=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError(secret_text)
            )
        )
    assert unexpected_error.value.category == "unexpected_error"
    assert secret_text not in render_llm_connectivity_error(unexpected_error.value)


def test_connectivity_check_fails_closed_on_response_model_mismatch(monkeypatch):
    import numpilot.llm_check as check

    content = LLM_CHECK_MARKER
    monkeypatch.setattr(check, "load_llm_config", lambda _path=None: _configured_status())
    monkeypatch.setattr(check, "llm_proxy_route_status", lambda: _proxy())

    with pytest.raises(LlmConnectivityCheckError) as caught:
        run_llm_connectivity_check(
            chat=lambda *_args, **_kwargs: (
                content,
                _receipt(content, response_model="other/model"),
            )
        )

    assert caught.value.category == "response_invalid"
    assert caught.value.actual_remote_call is True


def test_check_llm_command_has_stable_success_and_failure_exit_codes(monkeypatch):
    import numpilot.llm_check as check

    content = LLM_CHECK_MARKER
    monkeypatch.setattr(check, "load_llm_config", lambda _path=None: _configured_status())
    monkeypatch.setattr(check, "llm_proxy_route_status", lambda: _proxy())
    output = io.StringIO()
    errors = io.StringIO()
    assert run_llm_check_command(
        stdout=output,
        stderr=errors,
        chat=lambda *_args, **_kwargs: (content, _receipt(content)),
    ) == 0
    assert "可以开始迁移" in output.getvalue()
    assert errors.getvalue() == ""

    output = io.StringIO()
    errors = io.StringIO()
    assert run_llm_check_command(
        stdout=output,
        stderr=errors,
        chat=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("https://private-endpoint.invalid secret-token")
        ),
    ) == 4
    assert "private-endpoint" not in errors.getvalue()
    assert "secret-token" not in errors.getvalue()


def test_check_llm_command_colors_only_tty_human_output(monkeypatch):
    import numpilot.llm_check as check

    content = LLM_CHECK_MARKER
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("NUMPILOT_PLAIN", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(check, "load_llm_config", lambda _path=None: _configured_status())
    monkeypatch.setattr(check, "llm_proxy_route_status", lambda: _proxy())
    output = _TtyBuffer()
    errors = _TtyBuffer()

    assert run_llm_check_command(
        stdout=output,
        stderr=errors,
        chat=lambda *_args, **_kwargs: (content, _receipt(content)),
    ) == 0

    assert "\x1b[" in output.getvalue()
    assert "可以开始迁移" in _ANSI.sub("", output.getvalue())
    assert errors.getvalue() == ""
