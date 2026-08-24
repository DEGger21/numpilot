"""Safety regressions for the external Poisson qualification harness."""

import importlib.util
import os
from pathlib import Path
import sys

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "poisson_qualification_harness.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "poisson_qualification_harness", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
harness = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(harness)


def test_subprocess_environment_is_private_and_only_inherits_safe_device_list(
    tmp_path, monkeypatch
):
    sentinel = "NUMPILOT_PARENT_SECRET_MUST_NOT_CROSS"
    inherited_names = (
        "NUMPILOT_LLM_AUTH_TOKEN",
        "NUMPILOT_LLM_BASE_URL",
        "JPY_AUTH_TOKEN",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "SSH_AUTH_SOCK",
        "GIT_ASKPASS",
        "PYTHONSTARTUP",
        "LD_PRELOAD",
        "ASCEND_RT_VISIBLE_DEVICES",
        "PATH",
        "PYTHONPATH",
        "HOME",
        "TMPDIR",
        "XDG_CONFIG_HOME",
    )
    for name in inherited_names:
        monkeypatch.setenv(name, sentinel)
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", "2,5")

    private_opp = tmp_path / "private-opp"
    for config in harness.OPERATOR_CONFIG.values():
        (private_opp / "vendors" / config["vendor"]).mkdir(parents=True)
    sandbox = tmp_path / "sandbox"
    env = harness._command_env(
        sandbox_root=sandbox,
        python=Path(sys.executable),
        private_opp=private_opp,
    )

    assert env["ASCEND_VISIBLE_DEVICES"] == "2,5"
    assert env["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) == [
        str(private_opp / "vendors" / config["vendor"])
        for config in harness.OPERATOR_CONFIG.values()
    ]
    assert sentinel not in env.values()
    for name in inherited_names[:12]:
        assert name not in env
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(_SCRIPT.parents[1]),
        str(
            Path(sys.executable).absolute().parent.parent
            / "lib/python3.10/site-packages"
        ),
        str(
            _SCRIPT.parents[2]
            / "numpilot-gap-runs/asnumpy-runtime-20260813/venv/lib/"
            "python3.10/site-packages"
        ),
        "/usr/local/Ascend/cann-9.0.0/python/site-packages",
        "/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe",
    ]
    assert env["PATH"].split(os.pathsep)[0] == str(
        Path(sys.executable).absolute().parent
    )
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert env["TMP"] == env["TEMP"] == env["TMPDIR"]
    resolved_sandbox = sandbox.resolve()
    for name in (
        "HOME",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_RUNTIME_DIR",
        "ASNUMPY_LOG_DIR",
    ):
        path = Path(env[name])
        path.relative_to(resolved_sandbox)
        assert path.is_dir()
        assert path.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    "selection",
    ("0,0", " 0", "0, 1", "0;echo-leak", "-1", "00"),
)
def test_subprocess_environment_rejects_unsafe_device_selection(
    selection, tmp_path, monkeypatch
):
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", selection)

    with pytest.raises(RuntimeError, match="ASCEND_VISIBLE_DEVICES"):
        harness._command_env(
            sandbox_root=tmp_path / "sandbox",
            python=Path(sys.executable),
        )


def test_subprocess_environment_does_not_invent_a_device_selection(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ASCEND_VISIBLE_DEVICES", raising=False)

    env = harness._command_env(
        sandbox_root=tmp_path / "sandbox",
        python=Path(sys.executable),
    )

    assert "ASCEND_VISIBLE_DEVICES" not in env
    assert "ASCEND_CUSTOM_OPP_PATH" not in env


def test_virtualenv_python_symlink_is_not_resolved_before_execution(tmp_path):
    executable = tmp_path / "venv-python"
    executable.symlink_to(Path(sys.executable).resolve())

    selected = harness._require_python(str(executable))

    assert selected == executable.absolute()
    assert selected.is_symlink()


def test_pytest_skip_is_never_counted_as_a_qualification_pass():
    with pytest.raises(RuntimeError, match="non-pass"):
        harness._pytest_count("10 passed, 3 skipped in 0.41s", "example")

    assert harness._pytest_count("13 passed in 1.2s", "example") == 13


def test_private_opp_internal_symlink_is_materialized(tmp_path):
    target = tmp_path / "vendor" / "lib" / "libtarget.so"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bound-runtime-library")
    link = tmp_path / "vendor" / "liboptiling.so"
    link.symlink_to("lib/libtarget.so")

    harness._materialize_private_opp_symlinks(tmp_path)

    assert link.is_file()
    assert not link.is_symlink()
    assert link.read_bytes() == target.read_bytes()


def test_private_opp_symlink_may_not_escape_the_evidence_root(tmp_path):
    root = tmp_path / "opp"
    root.mkdir()
    outside = tmp_path / "outside.so"
    outside.write_bytes(b"outside")
    link = root / "liboptiling.so"
    link.symlink_to(outside)

    with pytest.raises(RuntimeError, match="escapes"):
        harness._materialize_private_opp_symlinks(root)

    assert link.is_symlink()
