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

"""Regression tests for the interactive source-interpretation boundary.

These tests deliberately inject the transport callback.  They exercise the
same receipt validation and source/path safety used by the terminal wrapper,
without making a network request or requiring an Ascend runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from numpilot.interactive_migration import (
    InteractiveMigrationError,
    MAX_MODEL_SOURCE_BYTES,
    SourceSnapshot,
    _visible_completion_options,
    build_interpretation_prompt,
    failure_report_payload,
    interpret_source_with_configured_model,
    parse_interpretation,
    read_source_snapshot,
    sanitize_interpretation,
    write_interpretation_report,
)
from numpilot.llm_config import (
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    model_identity_sha256,
    save_llm_config,
)


def _receipt(content: str, *, model: str = "glm-test", provider: str = "glm"):
    identity_hash = model_identity_sha256(
        protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        provider=provider,
        configured_model=model,
        requested_model=model,
        response_model=model,
        response_model_aliases=(),
        response_model_identity="exact",
    )
    return {
        "schema_version": 2,
        "utc_start": "2026-09-04T00:00:00Z",
        "utc_end": "2026-09-04T00:00:01Z",
        "http_status": 200,
        "requested_model": model,
        "response_model": model,
        "response_id": "response-test",
        "response_created": 1,
        "request_body_sha256": "1" * 64,
        "response_body_sha256": "2" * 64,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "request_id": "request-test",
        "usage": {"total_tokens": 8},
        "tls_verification": "enabled",
        "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        "provider": provider,
        "configured_model": model,
        "model_identity_sha256": identity_hash,
        "response_model_aliases": [],
        "response_model_identity": "exact",
    }


def test_glm_interpretation_prefers_visible_output_without_changing_other_providers():
    glm = SimpleNamespace(provider="glm", model="glm-5.3-flash")
    other = SimpleNamespace(provider="openai", model="example-model")

    assert _visible_completion_options(glm, 1800, force_visible=True) == {
        "max_tokens": 1800,
        "thinking_mode": "disabled",
        "reasoning_effort": "none",
    }
    assert _visible_completion_options(glm, 1800, force_visible=False) == {
        "max_tokens": 1800,
    }
    assert _visible_completion_options(other, 1800, force_visible=True) == {
        "max_tokens": 1800,
    }


def _config(tmp_path: Path) -> Path:
    return save_llm_config(
        "http://model.invalid/v1",
        "interactive-secret-token",
        "glm-test",
        provider="glm",
        path=str(tmp_path / "private" / "llm.json"),
    )


def _clear_llm_environment(monkeypatch):
    for name in (
        "NUMPILOT_LLM_CONFIG_FILE",
        "NUMPILOT_LLM_BASE_URL",
        "NUMPILOT_LLM_AUTH_TOKEN",
        "NUMPILOT_LLM_MODEL",
        "NUMPILOT_LLM_PROVIDER",
        "NUMPILOT_LLM_PROTOCOL",
        "NUMPILOT_LLM_RESPONSE_MODEL_ALIASES",
        "NUMPILOT_LLM_REQUIRE_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


def test_read_source_snapshot_requires_absolute_regular_utf8_and_hashes(tmp_path):
    source = tmp_path / "case.py"
    payload = "# demo\nresult = 1\n"
    source.write_text(payload, encoding="utf-8")

    snapshot = read_source_snapshot(source)

    assert isinstance(snapshot, SourceSnapshot)
    assert snapshot.path == str(source)
    assert snapshot.text == payload
    assert snapshot.sha256 == hashlib.sha256(payload.encode()).hexdigest()
    assert snapshot.bytes == len(payload.encode())
    assert snapshot.lines == 2

    with pytest.raises(InteractiveMigrationError, match="absolute"):
        read_source_snapshot("case.py")

    link = tmp_path / "link.py"
    link.symlink_to(source)
    with pytest.raises(InteractiveMigrationError, match="symlink"):
        read_source_snapshot(link)

    bad = tmp_path / "bad.py"
    bad.write_bytes(b"\xff\xfe")
    with pytest.raises(InteractiveMigrationError, match="UTF-8"):
        read_source_snapshot(bad)


def test_read_source_snapshot_enforces_bound_and_rejects_directory(tmp_path):
    source = tmp_path / "large.py"
    source.write_bytes(b"x" * 32)
    with pytest.raises(InteractiveMigrationError, match="exceeds"):
        read_source_snapshot(source, max_bytes=16)

    with pytest.raises(InteractiveMigrationError, match="regular"):
        read_source_snapshot(tmp_path)


def test_prompt_binds_full_source_hash_and_marks_large_prompt_truncation(tmp_path):
    source = tmp_path / "large.py"
    payload = "x = 1\n" * ((MAX_MODEL_SOURCE_BYTES // 6) + 100)
    source.write_text(payload, encoding="utf-8")
    snapshot = read_source_snapshot(source)

    system, user, truncated = build_interpretation_prompt(
        snapshot,
        static_analysis={
            "decision": "DIRECT",
            "qualified_apis": ["add"],
            "secret": "must-not-be-forwarded",
        },
    )

    assert "普通 NumPy 程序" in system
    assert snapshot.sha256 in user
    assert "<untrusted_source>" in user and "</untrusted_source>" in user
    assert "middle of source omitted" in user
    assert truncated is True
    assert "must-not-be-forwarded" not in user


def test_parse_and_sanitize_interpretation_redact_controls_and_credentials(
    monkeypatch,
):
    monkeypatch.setenv("NUMPILOT_LLM_AUTH_TOKEN", "interactive-secret-token")
    content = json.dumps({
        "summary": "\x1b[31mok\x1b[0m",
        "auth_token": "interactive-secret-token",
        "direction": "\u202e",
    })
    parsed = parse_interpretation(content)

    assert parsed.format == "json"
    assert parsed.fields is not None
    assert "interactive-secret-token" not in parsed.text
    assert "\x1b" not in parsed.text
    assert "interactive-secret-token" not in json.dumps(parsed.fields)
    assert "�" in parsed.text  # bidi format character is made visible safely

    bounded = sanitize_interpretation("a" * 100, max_chars=20)
    assert bounded.startswith("a" * 20)
    assert "truncated" in bounded


def test_interpret_source_uses_private_config_and_validated_callback_receipt(
    tmp_path, monkeypatch
):
    _clear_llm_environment(monkeypatch)
    source = tmp_path / "case.py"
    source.write_text("import numpy as np\nresult = np.exp(1)\n", encoding="utf-8")
    config = _config(tmp_path)
    content = "# 解读\n- 主要计算：逐元素指数\n- 性能：尚未测量"
    seen = {}

    def chat(system, user, *, max_tokens):
        seen["system"] = system
        seen["user"] = user
        seen["max_tokens"] = max_tokens
        return content, _receipt(content)

    result = interpret_source_with_configured_model(
        source,
        config_path=config,
        max_tokens=1800,
        chat=chat,
    )

    assert result.status == "interpreted_unverified"
    assert result.actual_remote_call is True
    assert result.model == "glm-test"
    assert result.provider == "glm"
    assert result.source_path == str(source)
    assert result.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result.display_format == "markdown"
    assert "主要计算" in result.display_text
    assert result.receipt_metadata["http_status"] == 200
    assert seen["max_tokens"] == 1800
    assert result.source_path in seen["user"]
    # The temporary config scope must not leak credentials into this process.
    assert os.environ.get("NUMPILOT_LLM_AUTH_TOKEN") is None

    payload = result.to_dict()
    assert "display_text" not in payload
    assert payload["display_text_persisted"] is False
    assert payload["remote_receipt"]["response_body_sha256"] == "2" * 64
    assert "interactive-secret-token" not in json.dumps(payload)


def test_explicit_config_path_does_not_normalize_away_symlink_component(
    tmp_path, monkeypatch
):
    """A ``link/..`` spelling must be rejected before a credential read."""

    _clear_llm_environment(monkeypatch)
    source = tmp_path / "case.py"
    source.write_text("result = 1\n", encoding="utf-8")
    config = _config(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "private", target_is_directory=True)

    # The normalized destination is the real config, but resolving the
    # symlink component first would let a caller redirect the intended path.
    disguised = link / ".." / "private" / "llm.json"
    with pytest.raises(InteractiveMigrationError, match="regular file|symlink"):
        interpret_source_with_configured_model(
            source,
            config_path=disguised,
            chat=lambda *_args, **_kwargs: pytest.fail("must not call model"),
        )


def test_interpret_source_retries_once_only_for_empty_visible_content(tmp_path, monkeypatch):
    _clear_llm_environment(monkeypatch)
    source = tmp_path / "case.py"
    source.write_text("result = 1\n", encoding="utf-8")
    config = _config(tmp_path)
    calls = []

    def chat(_system, _user, *, max_tokens):
        calls.append(max_tokens)
        content = "" if len(calls) == 1 else "# 可迁移算子\n- 尚未验证"
        return content, _receipt(content)

    result = interpret_source_with_configured_model(
        source, config_path=config, max_tokens=1800, chat=chat
    )

    assert calls == [1800, 3600]
    assert result.receipt_metadata["interpretation_attempts"] == 2
    assert result.receipt_metadata["initial_max_tokens"] == 1800
    assert result.receipt_metadata["final_max_tokens"] == 3600


def test_interpretation_rejects_invalid_receipt_before_returning_claim(
    tmp_path, monkeypatch
):
    _clear_llm_environment(monkeypatch)
    source = tmp_path / "case.py"
    source.write_text("result = 1\n", encoding="utf-8")
    config = _config(tmp_path)
    called = {"count": 0}

    def chat(_system, _user, **_kwargs):
        called["count"] += 1
        return "ok", {"schema_version": 2}

    with pytest.raises(InteractiveMigrationError, match="receipt") as caught:
        interpret_source_with_configured_model(source, config_path=config, chat=chat)

    assert called["count"] == 1
    assert caught.value.actual_remote_call is True


def test_write_interpretation_report_is_private_metadata_only(tmp_path, monkeypatch):
    _clear_llm_environment(monkeypatch)
    source = tmp_path / "case.py"
    source.write_text("result = 1\n", encoding="utf-8")
    config = _config(tmp_path)
    content = "解读中包含不应落盘的文本和 interactive-secret-token"

    result = interpret_source_with_configured_model(
        source,
        config_path=config,
        chat=lambda *_args, **_kwargs: (content, _receipt(content)),
    )
    report = write_interpretation_report(tmp_path / "receipts" / "interpretation.json", result)

    assert report.is_absolute()
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    assert stat.S_IMODE(report.parent.stat().st_mode) == 0o700
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "interpreted_unverified"
    assert data["display_text_persisted"] is False
    assert "display_text" not in data
    assert content not in report.read_text(encoding="utf-8")
    with pytest.raises(InteractiveMigrationError, match="already exists"):
        write_interpretation_report(report, result)


def test_failure_report_is_bounded_and_redacts_error_text(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMPILOT_LLM_AUTH_TOKEN", "interactive-secret-token")
    source = tmp_path / "case.py"
    source.write_text("result = 1\n", encoding="utf-8")
    snapshot = read_source_snapshot(source)
    error = InteractiveMigrationError(
        "provider failed Bearer interactive-secret-token",
        actual_remote_call=True,
    )

    payload = failure_report_payload(snapshot, error, elapsed_ms=12)

    assert payload["status"] == "failed"
    assert payload["actual_remote_call"] is True
    assert payload["elapsed_ms"] == 12
    assert "interactive-secret-token" not in json.dumps(payload)
    assert "Bearer [redacted]" in payload["error_detail"]
