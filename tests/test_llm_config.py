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
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from numpilot import __main__ as cli
from numpilot import llm_config as llm_config_module
from numpilot.llm_config import (
    LlmConfigError,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    configure_llm_interactive,
    init_llm_config_template,
    llm_config_path,
    load_llm_config,
    save_llm_config,
    model_identity_sha256,
)


_LLM_ENV = (
    "NUMPILOT_LLM_BASE_URL",
    "NUMPILOT_LLM_AUTH_TOKEN",
    "NUMPILOT_LLM_MODEL",
    "NUMPILOT_LLM_PROVIDER",
    "NUMPILOT_LLM_CONFIG_FILE",
    "NUMPILOT_LLM_PROTOCOL",
    "NUMPILOT_LLM_RESPONSE_MODEL_ALIASES",
)


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    for key in _LLM_ENV:
        monkeypatch.delenv(key, raising=False)
    config_home = tmp_path / "config"
    monkeypatch.setenv("NUMPILOT_CONFIG_HOME", str(config_home))
    return config_home / "llm.json"


def test_save_and_load_private_config(isolated_config, monkeypatch):
    secret = "sentinel-secret"
    path = save_llm_config("https://gateway.example/v1", secret, "glm-test")

    assert path == isolated_config
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700

    for key in _LLM_ENV[:4]:
        monkeypatch.delenv(key, raising=False)
    status = load_llm_config()
    assert status.configured
    assert status.source == "user-config"
    assert os.environ["NUMPILOT_LLM_AUTH_TOKEN"] == secret
    assert secret not in status.summary()
    assert secret not in repr(status)


def test_init_template_is_private_empty_and_manually_editable(isolated_config):
    path, created = init_llm_config_template()

    assert created
    assert path == isolated_config
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 2,
        "base_url": "",
        "auth_token": "",
        "model": "",
        "provider": "openai-compatible",
        "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        "response_model_aliases": [],
    }
    status = load_llm_config()
    assert not status.configured
    assert status.missing == (
        "NUMPILOT_LLM_BASE_URL",
        "NUMPILOT_LLM_AUTH_TOKEN",
        "NUMPILOT_LLM_MODEL",
    )


def test_init_template_never_overwrites_existing_config(isolated_config):
    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text('{"user": "content"}\n', encoding="utf-8")
    before = isolated_config.read_bytes()

    path, created = init_llm_config_template()

    assert path == isolated_config
    assert not created
    assert isolated_config.read_bytes() == before


def test_config_read_and_write_reject_symlinked_parent(isolated_config, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = isolated_config.parent.parent / "linked-config"
    linked.symlink_to(outside, target_is_directory=True)
    target = linked / "llm.json"

    with pytest.raises(LlmConfigError, match="符号链接"):
        save_llm_config(
            "https://gateway.example/v1", "secret", "model", path=str(target)
        )
    with pytest.raises(LlmConfigError, match="符号链接"):
        load_llm_config(path=str(target))
    assert not (outside / "llm.json").exists()


def test_config_rejects_dotdot_path_that_crosses_symlink(isolated_config, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = isolated_config.parent.parent / "linked-config"
    linked.symlink_to(outside, target_is_directory=True)
    target = linked / ".." / "escaped" / "llm.json"

    # Normalizing this spelling would erase ``linked``; filesystem resolution
    # follows it first, so the config boundary must reject the path before any
    # mkdir/write operation can occur.
    with pytest.raises(LlmConfigError, match="符号链接"):
        save_llm_config(
            "https://gateway.example/v1", "secret", "model", path=str(target)
        )
    assert not (outside / "escaped" / "llm.json").exists()


def test_config_read_rejects_world_or_group_accessible_parent(isolated_config):
    isolated_config.parent.mkdir(parents=True)
    isolated_config.parent.chmod(0o755)
    isolated_config.write_text(
        json.dumps(
            {
                "base_url": "https://gateway.example/v1",
                "auth_token": "secret",
                "model": "model",
            }
        ),
        encoding="utf-8",
    )
    isolated_config.chmod(0o600)

    with pytest.raises(LlmConfigError, match="目录权限过宽"):
        load_llm_config(path=str(isolated_config))


def test_environment_overrides_file_field_by_field(isolated_config, monkeypatch):
    save_llm_config("https://file.example/v1", "file-token", "file-model")
    monkeypatch.setenv("NUMPILOT_LLM_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("NUMPILOT_LLM_MODEL", "   ")

    status = load_llm_config()

    assert status.configured
    assert status.source == "environment+user-config"
    assert os.environ["NUMPILOT_LLM_BASE_URL"] == "https://env.example/v1"
    assert os.environ["NUMPILOT_LLM_AUTH_TOKEN"] == "file-token"
    assert os.environ["NUMPILOT_LLM_MODEL"] == "file-model"


def test_load_rolls_back_environment_when_saved_validation_fails(
    isolated_config, monkeypatch
):
    isolated_config.parent.mkdir(parents=True)
    isolated_config.parent.chmod(0o700)
    isolated_config.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "base_url": "https://gateway.example/v1",
                "auth_token": "should-not-linger",
                "model": "org/model",
                "provider": "invalid provider!",
                "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                "response_model_aliases": [],
            }
        ),
        encoding="utf-8",
    )
    isolated_config.chmod(0o600)

    with pytest.raises(LlmConfigError):
        load_llm_config()

    for key in _LLM_ENV[:4] + (_LLM_ENV[5], _LLM_ENV[6]):
        assert key not in os.environ


def test_environment_model_override_does_not_inherit_saved_alias_policy(
    isolated_config, monkeypatch
):
    save_llm_config(
        "https://file.example/v1",
        "file-token",
        "file/model",
        response_model_aliases=("file/alias",),
    )
    monkeypatch.setenv("NUMPILOT_LLM_MODEL", "env/model")

    status = load_llm_config()

    assert status.model == "env/model"
    assert status.response_model_aliases == ()
    assert "NUMPILOT_LLM_RESPONSE_MODEL_ALIASES" not in os.environ


def test_interactive_model_change_drops_old_aliases_unless_explicit(
    isolated_config, monkeypatch
):
    save_llm_config(
        "https://gateway.example/v1",
        "saved-token",
        "old/model",
        response_model_aliases=("old/alias",),
    )
    for key in _LLM_ENV[:4]:
        monkeypatch.delenv(key, raising=False)

    status = configure_llm_interactive(
        base_url="https://gateway.example/v1",
        model="new/model",
        response_model_aliases=None,
        secret_fn=lambda _prompt: "",
    )

    assert status.response_model_aliases == ()
    saved = json.loads(isolated_config.read_text(encoding="utf-8"))
    assert saved["model"] == "new/model"
    assert saved["response_model_aliases"] == []


def test_complete_environment_bypasses_corrupt_file(isolated_config, monkeypatch):
    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text("{broken", encoding="utf-8")
    for key, value in {
        "NUMPILOT_LLM_BASE_URL": "https://env.example/v1",
        "NUMPILOT_LLM_AUTH_TOKEN": "env-token",
        "NUMPILOT_LLM_MODEL": "env-model",
    }.items():
        monkeypatch.setenv(key, value)

    status = load_llm_config()

    assert status.configured
    assert status.source == "environment"


@pytest.mark.parametrize("content", ["{broken", "[]", '{"base_url": 3}'])
def test_malformed_config_fails_without_leaking_content(isolated_config, content):
    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text(content, encoding="utf-8")
    isolated_config.chmod(0o600)

    with pytest.raises(LlmConfigError) as caught:
        load_llm_config()

    assert content not in str(caught.value)


@pytest.mark.parametrize(
    "field, value",
    (
        ("base_url", None),
        ("auth_token", 123),
        ("model", object()),
        ("provider", ["gateway"]),
    ),
)
def test_save_rejects_non_string_credentials_and_labels(
    isolated_config, field, value
):
    arguments = {
        "base_url": "https://gateway.example/v1",
        "auth_token": "secret",
        "model": "org/model",
        "provider": "gateway",
    }
    arguments[field] = value

    with pytest.raises(LlmConfigError):
        save_llm_config(**arguments)
    assert not isolated_config.exists()


def test_read_rejects_oversized_saved_config_without_unbounded_load(isolated_config):
    isolated_config.parent.mkdir(parents=True)
    isolated_config.parent.chmod(0o700)
    isolated_config.write_bytes(b"{" + b"x" * (1024 * 1024) + b"}")
    isolated_config.chmod(0o600)

    with pytest.raises(LlmConfigError, match="过大"):
        load_llm_config()


def test_save_rejects_oversized_base_url_before_writing(isolated_config):
    oversized = "https://gateway.example/" + "x" * 4096

    with pytest.raises(LlmConfigError, match="BASE_URL"):
        save_llm_config(oversized, "secret", "org/model")

    assert not isolated_config.exists()


def test_manually_edited_model_identity_rejects_boundary_whitespace(
    isolated_config
):
    isolated_config.parent.mkdir(parents=True)
    # Keep the manually-authored fixture inside the private-directory
    # contract so this test reaches model-field validation.
    isolated_config.parent.chmod(0o700)
    isolated_config.write_text(
        json.dumps({
            "schema_version": 2,
            "base_url": "https://gateway.example/v1",
            "auth_token": "fake-private-token",
            "model": " org/model ",
            "provider": "gateway",
            "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            "response_model_aliases": [],
        }),
        encoding="utf-8",
    )
    isolated_config.chmod(0o600)

    with pytest.raises(LlmConfigError, match="model.*两端空白"):
        load_llm_config()


def test_wide_permissions_fail_closed(isolated_config):
    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text(
        json.dumps({
            "base_url": "https://gateway.example/v1",
            "auth_token": "never-print-this",
            "model": "glm-test",
        }),
        encoding="utf-8",
    )
    isolated_config.chmod(0o644)

    with pytest.raises(LlmConfigError) as caught:
        load_llm_config()

    assert "权限过宽" in str(caught.value)
    assert "never-print-this" not in str(caught.value)


def test_interactive_update_keeps_hidden_existing_token(isolated_config, monkeypatch):
    save_llm_config("https://old.example/v1", "old-secret", "old-model")
    for key in _LLM_ENV[:4]:
        monkeypatch.delenv(key, raising=False)
    prompts = []

    status = configure_llm_interactive(
        base_url="https://new.example/v1",
        model="new-model",
        secret_fn=lambda prompt: prompts.append(prompt) or "",
    )

    assert status.configured
    assert all("old-secret" not in prompt for prompt in prompts)
    saved = json.loads(isolated_config.read_text(encoding="utf-8"))
    assert saved["auth_token"] == "old-secret"
    assert saved["base_url"] == "https://new.example/v1"


@pytest.mark.parametrize("field", ("provider", "protocol"))
def test_interactive_rejects_non_string_labels_without_attribute_error(
    isolated_config, field
):
    arguments = {
        "base_url": "https://gateway.example/v1",
        "model": "org/model",
        "auth_token": "secret",
        field: 123,
    }

    with pytest.raises(LlmConfigError, match="字符串"):
        configure_llm_interactive(path=str(isolated_config), **arguments)

    assert not isolated_config.exists()


def test_atomic_save_failure_preserves_previous_config(isolated_config, monkeypatch):
    save_llm_config("https://old.example/v1", "old-token", "old-model")
    before = isolated_config.read_bytes()

    def fail_replace(_source, _target):
        raise OSError("simulated")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(LlmConfigError):
        save_llm_config("https://new.example/v1", "new-token", "new-model")
    assert isolated_config.read_bytes() == before


@pytest.mark.parametrize(
    ("base_url", "token"),
    (
        ("https://gateway.example:bad/v1", "token"),
        ("https://gateway.example:70000/v1", "token"),
        ("https://[broken/v1", "token"),
        ("https://user:pass@gateway.example/v1", "token"),
        ("https://gateway.example/v1?token=value", "token"),
        ("https://gateway.example/v1", "token with space"),
        ("https://gateway.example/v1", "token\x01control"),
        ("https://gateway.example/v1", "token\u202eformat"),
        ("https://gateway.example/v1\u202e", "token"),
    ),
)
def test_local_readiness_rejects_transport_values_before_saving(
    isolated_config, base_url, token
):
    with pytest.raises(LlmConfigError):
        save_llm_config(base_url, token, "org/model")
    assert not isolated_config.exists()


def test_cli_token_stdin_and_check_never_echo_token(isolated_config, monkeypatch, capsys):
    secret = "cli-sentinel-secret"
    monkeypatch.setattr(sys, "stdin", io.StringIO(secret + "\n"))

    rc = cli.main([
        "configure-llm",
        "--base-url", "https://gateway.example/v1",
        "--model", "glm-test",
        "--token-stdin",
    ])
    first = capsys.readouterr()
    assert rc == 0
    assert secret not in first.out + first.err

    for key in _LLM_ENV[:4]:
        monkeypatch.delenv(key, raising=False)
    rc = cli.main(["configure-llm", "--check"])
    second = capsys.readouterr()
    assert rc == 0
    assert secret not in second.out + second.err


def test_cli_check_is_local_only_and_never_calls_remote(
    isolated_config, monkeypatch, capsys
):
    save_llm_config("https://gateway.example/v1", "saved-secret", "glm-test")
    for key in _LLM_ENV[:4]:
        monkeypatch.delenv(key, raising=False)

    def forbidden_remote_call(_system, _user):
        raise AssertionError("--check must not make a remote request")

    monkeypatch.setattr(
        cli.LlmEngine,
        "_chat_with_receipt",
        staticmethod(forbidden_remote_call),
    )

    rc = cli.main(["configure-llm", "--check"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "LLM 已配置" in captured.out


def _probe_receipt(response_body):
    return {
        "schema_version": 2,
        "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        "provider": "gateway",
        "configured_model": "org/model",
        "http_status": 200,
        "tls_verification": "enabled",
        "requested_model": "org/model",
        "response_model": "org/model",
        "response_model_aliases": [],
        "response_model_identity": "exact",
        "model_identity_sha256": model_identity_sha256(
            protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            provider="gateway",
            configured_model="org/model",
            requested_model="org/model",
            response_model="org/model",
            response_model_aliases=(),
            response_model_identity="exact",
        ),
        "content_sha256": hashlib.sha256(
            response_body.encode("utf-8")
        ).hexdigest(),
        "utc_start": "2026-08-26T00:00:00Z",
        "utc_end": "2026-08-26T00:00:01Z",
        "response_id": "fake-response",
        "response_created": 1787702400,
        "request_body_sha256": "1" * 64,
        "response_body_sha256": "2" * 64,
        "request_id": "fake-request",
        "usage": {"total_tokens": 9},
    }


def test_cli_probe_reports_verified_remote_call_without_response_body_or_token(
    isolated_config, monkeypatch, capsys
):
    secret = "probe-sentinel-secret"
    response_body = "NUMPILOT_LLM_READY private model response must not be printed"
    save_llm_config(
        "https://gateway.example/v1", secret, "org/model", provider="gateway"
    )
    for key in _LLM_ENV[:4]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(
        cli.LlmEngine,
        "_chat_with_receipt",
        staticmethod(
            lambda _system, _user: (response_body, _probe_receipt(response_body))
        ),
    )

    rc = cli.main(["configure-llm", "--probe"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert rc == 0
    assert "actual_remote_call=yes" in captured.out
    assert "http_status=200" in captured.out
    assert "requested_model=org/model" in captured.out
    assert "response_model=org/model" in captured.out
    assert response_body not in output
    assert secret not in output


def test_cli_probe_rejects_valid_receipt_without_readiness_marker(
    isolated_config, monkeypatch, capsys
):
    response_body = "valid but not the required acknowledgement"
    save_llm_config(
        "https://gateway.example/v1",
        "marker-test-secret",
        "org/model",
        provider="gateway",
    )
    for key in _LLM_ENV[:4]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        cli.LlmEngine,
        "_chat_with_receipt",
        staticmethod(
            lambda _system, _user: (response_body, _probe_receipt(response_body))
        ),
    )

    rc = cli.main(["configure-llm", "--probe"])

    captured = capsys.readouterr()
    assert rc == 4
    assert "actual_remote_call=yes" not in captured.out + captured.err
    assert response_body not in captured.out + captured.err


def test_cli_probe_failure_is_explicit_and_nonzero(
    isolated_config, monkeypatch, capsys
):
    secret = "probe-failure-secret"
    save_llm_config("https://gateway.example/v1", secret, "glm-requested")
    for key in _LLM_ENV[:4]:
        monkeypatch.delenv(key, raising=False)

    def fail_remote_call(_system, _user):
        raise cli.LlmHttpError(503)

    monkeypatch.setattr(
        cli.LlmEngine,
        "_chat_with_receipt",
        staticmethod(fail_remote_call),
    )

    rc = cli.main(["configure-llm", "--probe"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert rc == 4
    assert "失败" in captured.err
    assert "HTTP status 503" in captured.err
    assert "actual_remote_call=yes" not in output
    assert secret not in output


def test_cli_probe_rejects_content_hash_mismatch(
    isolated_config, monkeypatch, capsys
):
    save_llm_config("https://gateway.example/v1", "saved-secret", "glm-requested")
    for key in _LLM_ENV[:4]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        cli.LlmEngine,
        "_chat_with_receipt",
        staticmethod(lambda _system, _user: (
            "real content",
            {
                "schema_version": 1,
                "http_status": 200,
                "tls_verification": "enabled",
                "requested_model": "glm-requested",
                "response_model": "glm-response",
                "content_sha256": "0" * 64,
                "usage": {},
            },
        )),
    )

    rc = cli.main(["configure-llm", "--probe"])

    captured = capsys.readouterr()
    assert rc == 4
    assert "回执校验不一致" in captured.err
    assert "actual_remote_call=yes" not in captured.out + captured.err


def test_noninteractive_config_refuses_to_prompt(isolated_config, monkeypatch, capsys):
    class NonTty(io.StringIO):
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", NonTty())
    rc = cli.main(["configure-llm"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "不是交互终端" in captured.err
    assert "--init" in captured.err


def test_cli_init_works_without_tty(isolated_config, monkeypatch, capsys):
    class NonTty(io.StringIO):
        def isatty(self):
            return False

    stdin = NonTty("must-not-be-read")
    monkeypatch.setattr(sys, "stdin", stdin)

    rc = cli.main(["configure-llm", "--init"])

    captured = capsys.readouterr()
    assert rc == 0
    assert isolated_config.is_file()
    assert str(isolated_config) in captured.out
    assert stdin.tell() == 0


def test_cli_init_rejects_inline_configuration(isolated_config, capsys):
    rc = cli.main([
        "configure-llm", "--init", "--base-url", "https://gateway.example/v1"
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "--init" in captured.err
    assert not isolated_config.exists()


@pytest.mark.parametrize(
    "arguments_prefix",
    (
        ("--llm-config",),
        (),
    ),
)
def test_cli_accepts_explicit_config_path_before_or_after_subcommand(
    isolated_config, monkeypatch, capsys, arguments_prefix
):
    explicit = isolated_config.parent / "external-runtime" / "llm.json"
    secret = "explicit-config-secret"
    save_llm_config(
        "https://gateway.example/v1",
        secret,
        "glm-explicit",
        path=str(explicit),
    )
    # The root option form is ``--llm-config FILE configure-llm --check``;
    # the child form is ``configure-llm --check --llm-config FILE``.
    if arguments_prefix:
        arguments = [*arguments_prefix, str(explicit), "configure-llm", "--check"]
    else:
        arguments = [
            "configure-llm",
            "--check",
            "--llm-config",
            str(explicit),
        ]

    try:
        rc = cli.main(arguments)

        captured = capsys.readouterr()
        assert rc == 0
        assert str(explicit) in captured.out
        assert secret not in captured.out + captured.err
        assert os.environ["NUMPILOT_LLM_MODEL"] == "glm-explicit"
    finally:
        # ``main`` is also a library entry point in the test suite.  Clear the
        # one-shot override so a later in-process command cannot inherit it.
        os.environ.pop("NUMPILOT_LLM_CONFIG_FILE", None)


def test_cli_llm_config_override_is_scoped_to_one_in_process_invocation(
    isolated_config, monkeypatch, capsys
):
    explicit = isolated_config.parent / "one-shot" / "llm.json"
    save_llm_config(
        "https://gateway.example/v1",
        "one-shot-secret",
        "glm-one-shot",
        path=str(explicit),
    )
    inherited = str(isolated_config.parent / "inherited.json")
    monkeypatch.setenv("NUMPILOT_LLM_CONFIG_FILE", inherited)

    rc = cli.main([
        "--llm-config",
        str(explicit),
        "configure-llm",
        "--check",
    ])

    capsys.readouterr()
    assert rc == 0
    assert os.environ["NUMPILOT_LLM_CONFIG_FILE"] == inherited


def test_llm_cli_delegates_configuration_loading_to_migrate_handler(
    isolated_config, monkeypatch
):
    save_llm_config("https://gateway.example/v1", "saved-token", "saved-model")
    for key in _LLM_ENV[:4]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cli, "_cmd_migrate", lambda _args: 0)

    rc = cli.main(["migrate", "example.py", "--engine", "llm"])

    assert rc == 0
    assert "NUMPILOT_LLM_AUTH_TOKEN" not in os.environ


def test_config_path_does_not_depend_on_current_directory(isolated_config, tmp_path, monkeypatch):
    expected = llm_config_path()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert llm_config_path() == expected


@pytest.mark.parametrize(
    "bad_path",
    ("", "bad\npath", "bad\rpath", "bad\x00path", "bad\u202epath"),
)
def test_explicit_config_path_rejects_control_or_empty_values(
    isolated_config, bad_path
):
    with pytest.raises(LlmConfigError, match="配置路径"):
        llm_config_path(bad_path)


@pytest.mark.parametrize(
    "bad_home",
    (
        "bad\nconfig",
        "bad\x1bconfig",
        "bad\u202econfig",
        "x" * 5000,
    ),
)
def test_environment_config_home_is_validated_before_path_use(
    isolated_config, monkeypatch, bad_home
):
    monkeypatch.setenv("NUMPILOT_CONFIG_HOME", bad_home)

    with pytest.raises(LlmConfigError, match="配置路径"):
        llm_config_path()


def test_run_demo_does_not_source_configuration():
    script = Path("run_demo.sh").read_text(encoding="utf-8")
    assert ".env.llm" not in script
    shell_lines = [line.strip() for line in script.splitlines()]
    assert not any(line.startswith("source ") or line.startswith(". ") for line in shell_lines)
    assert "configure-llm --check" in script


def _clear_checkout_config_environment(monkeypatch):
    """Make a checkout-local discovery test independent of the caller's env."""

    for key in _LLM_ENV:
        monkeypatch.delenv(key, raising=False)
    for key in (
        "NUMPILOT_HOME",
        "NUMPILOT_RUNTIME_ROOT",
        "NUMPILOT_CONFIG_HOME",
        "XDG_CONFIG_HOME",
        "NUMPILOT_DISABLE_REPO_CONFIG",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_checkout(tmp_path, *, token="checkout-secret"):
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'numpilot-test'\n", encoding="utf-8"
    )
    target = repo / "llm.json"
    target.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "base_url": "https://gateway.example/v1",
                "auth_token": token,
                "model": "glm-checkout",
                "provider": "openai-compatible",
                "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                "response_model_aliases": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(target, 0o600)
    return repo, target


def test_shipped_llm_example_is_blank_and_matches_schema():
    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads((repo_root / "llm.example.json").read_text(encoding="utf-8"))

    assert payload == {
        "schema_version": 2,
        "base_url": "",
        "auth_token": "",
        "model": "",
        "provider": "openai-compatible",
        "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        "response_model_aliases": [],
    }
    assert payload["auth_token"] == ""


def test_checkout_local_llm_json_is_discovered_without_explicit_home(
    tmp_path, monkeypatch
):
    _clear_checkout_config_environment(monkeypatch)
    repo, target = _make_checkout(tmp_path)
    os.chmod(repo, 0o750)
    monkeypatch.setattr(llm_config_module, "_repo_root", lambda: repo)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert llm_config_path() == target
    status = load_llm_config()

    assert status.configured
    assert status.path == target
    assert status.source == "user-config"
    assert os.environ["NUMPILOT_LLM_AUTH_TOKEN"] == "checkout-secret"


def test_checkout_local_llm_json_requires_private_file_mode(tmp_path, monkeypatch):
    _clear_checkout_config_environment(monkeypatch)
    repo, target = _make_checkout(tmp_path)
    os.chmod(repo, 0o750)
    os.chmod(target, 0o644)
    monkeypatch.setattr(llm_config_module, "_repo_root", lambda: repo)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    with pytest.raises(LlmConfigError, match="chmod 600"):
        load_llm_config()


def test_checkout_local_llm_json_rejects_writable_checkout(tmp_path, monkeypatch):
    _clear_checkout_config_environment(monkeypatch)
    repo, _target = _make_checkout(tmp_path)
    os.chmod(repo, 0o775)
    monkeypatch.setattr(llm_config_module, "_repo_root", lambda: repo)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    with pytest.raises(LlmConfigError, match="可写"):
        load_llm_config()


def test_explicit_config_home_wins_over_checkout_local(tmp_path, monkeypatch):
    _clear_checkout_config_environment(monkeypatch)
    repo, _target = _make_checkout(tmp_path)
    explicit_home = tmp_path / "explicit-config"
    monkeypatch.setattr(llm_config_module, "_repo_root", lambda: repo)
    monkeypatch.setenv("NUMPILOT_CONFIG_HOME", str(explicit_home))

    assert llm_config_path() == explicit_home / "llm.json"


def test_xdg_config_home_does_not_hide_checkout_local(tmp_path, monkeypatch):
    _clear_checkout_config_environment(monkeypatch)
    repo, target = _make_checkout(tmp_path)
    monkeypatch.setattr(llm_config_module, "_repo_root", lambda: repo)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))

    assert llm_config_path() == target


def test_disable_repo_config_skips_checkout_local(tmp_path, monkeypatch):
    _clear_checkout_config_environment(monkeypatch)
    repo, target = _make_checkout(tmp_path)
    monkeypatch.setattr(llm_config_module, "_repo_root", lambda: repo)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NUMPILOT_DISABLE_REPO_CONFIG", "1")

    assert llm_config_path() != target


def test_saving_checkout_local_config_preserves_checkout_mode(tmp_path, monkeypatch):
    _clear_checkout_config_environment(monkeypatch)
    repo, target = _make_checkout(tmp_path, token="old-secret")
    os.chmod(repo, 0o750)
    monkeypatch.setattr(llm_config_module, "_repo_root", lambda: repo)

    saved = save_llm_config(
        "https://gateway.example/v1",
        "new-secret",
        "glm-checkout",
        path=str(target),
    )

    assert saved == target
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(repo.stat().st_mode) == 0o750


def test_llm_json_is_ignored_and_example_is_tracked():
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / ".git").exists():
        pytest.skip("not a git checkout")

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "llm.json"],
        cwd=repo_root,
        check=False,
    )
    assert ignored.returncode == 0
    example_ignored = subprocess.run(
        ["git", "check-ignore", "-q", "llm.example.json"],
        cwd=repo_root,
        check=False,
    )
    assert example_ignored.returncode == 1
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "llm.example.json"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    assert tracked.returncode == 0


def test_no_tracked_llm_json_contains_a_token():
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / ".git").exists():
        pytest.skip("not a git checkout")

    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    for relative in listing.stdout.decode("utf-8").split("\0"):
        if not relative:
            continue
        if relative != "llm.json" and not relative.endswith("/llm.json"):
            continue
        payload = json.loads((repo_root / relative).read_text(encoding="utf-8"))
        assert payload.get("auth_token", "") == "", relative
