import hashlib
import io
import json
import os
from pathlib import Path
import sys

import pytest

from numpilot import __main__ as cli
from numpilot.llm_config import (
    LlmConfigError,
    configure_llm_interactive,
    init_llm_config_template,
    llm_config_path,
    load_llm_config,
    save_llm_config,
)


_LLM_ENV = (
    "NUMPILOT_LLM_BASE_URL",
    "NUMPILOT_LLM_AUTH_TOKEN",
    "NUMPILOT_LLM_MODEL",
    "NUMPILOT_LLM_PROVIDER",
    "NUMPILOT_LLM_CONFIG_FILE",
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
        "schema_version": 1,
        "base_url": "",
        "auth_token": "",
        "model": "",
        "provider": "glm",
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


def test_atomic_save_failure_preserves_previous_config(isolated_config, monkeypatch):
    save_llm_config("https://old.example/v1", "old-token", "old-model")
    before = isolated_config.read_bytes()

    def fail_replace(_source, _target):
        raise OSError("simulated")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(LlmConfigError):
        save_llm_config("https://new.example/v1", "new-token", "new-model")
    assert isolated_config.read_bytes() == before


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


def test_cli_probe_reports_verified_remote_call_without_response_body_or_token(
    isolated_config, monkeypatch, capsys
):
    secret = "probe-sentinel-secret"
    response_body = "private model response must not be printed"
    save_llm_config("https://gateway.example/v1", secret, "glm-requested")
    for key in _LLM_ENV[:4]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(
        cli.LlmEngine,
        "_chat_with_receipt",
        staticmethod(lambda _system, _user: (
            response_body,
            {
                "schema_version": 1,
                "http_status": 200,
                "tls_verification": "enabled",
                "requested_model": "glm-requested",
                "response_model": "glm-response",
                "content_sha256": hashlib.sha256(
                    response_body.encode("utf-8")
                ).hexdigest(),
                "usage": {"total_tokens": 9},
            },
        )),
    )

    rc = cli.main(["configure-llm", "--probe"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert rc == 0
    assert "actual_remote_call=yes" in captured.out
    assert "http_status=200" in captured.out
    assert "requested_model=glm-requested" in captured.out
    assert "response_model=glm-response" in captured.out
    assert response_body not in output
    assert secret not in output


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


def test_llm_cli_loads_saved_config_without_source(isolated_config, monkeypatch):
    save_llm_config("https://gateway.example/v1", "saved-token", "saved-model")
    for key in _LLM_ENV[:4]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cli, "_cmd_migrate", lambda _args: 0)

    rc = cli.main(["migrate", "example.py", "--engine", "llm"])

    assert rc == 0
    assert os.environ["NUMPILOT_LLM_AUTH_TOKEN"] == "saved-token"


def test_config_path_does_not_depend_on_current_directory(isolated_config, tmp_path, monkeypatch):
    expected = llm_config_path()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert llm_config_path() == expected


def test_run_demo_does_not_source_configuration():
    script = Path("run_demo.sh").read_text(encoding="utf-8")
    assert ".env.llm" not in script
    shell_lines = [line.strip() for line in script.splitlines()]
    assert not any(line.startswith("source ") or line.startswith(". ") for line in shell_lines)
    assert "configure-llm --check" in script
