import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys

import pytest

import numpilot.bootstrap as bootstrap
from numpilot.bootstrap import ASNUMPY_COMMIT, default_asnumpy_checkout
from numpilot.cannbot_skills import CANNBOT_COMMIT, default_skill_checkout
from numpilot.llm_config import llm_config_path
from numpilot.paths import default_bootstrap_receipt, state_home


_PATH_ENVIRONMENT = (
    "NUMPILOT_HOME",
    "NUMPILOT_STATE_HOME",
    "NUMPILOT_CONFIG_HOME",
    "NUMPILOT_CACHE_HOME",
    "NUMPILOT_BOOTSTRAP_RECEIPT",
    "NUMPILOT_ASNUMPY_CACHE",
    "NUMPILOT_CANNBOT_CACHE",
    "NUMPILOT_CANNBOT_SKILLS",
    "NUMPILOT_LLM_CONFIG_FILE",
    "XDG_STATE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
)


def _clear_path_environment(monkeypatch):
    for name in _PATH_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def _resolved_user_paths():
    return {
        "state": state_home(),
        "llm_config": llm_config_path(),
        "receipt": default_bootstrap_receipt(),
        "asnumpy": default_asnumpy_checkout(),
        "cannbot": default_skill_checkout(),
    }


def test_specific_path_variables_override_persistent_root_xdg_and_home(
    monkeypatch, tmp_path
):
    _clear_path_environment(monkeypatch)
    home = tmp_path / "transient-home"
    root = tmp_path / "persistent"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("NUMPILOT_HOME", str(root))

    state = tmp_path / "specific-state"
    config = tmp_path / "specific-config"
    receipt = tmp_path / "specific-receipts" / "ready.json"
    asnumpy_base = tmp_path / "specific-asnumpy"
    cannbot_base = tmp_path / "specific-cannbot"
    monkeypatch.setenv("NUMPILOT_STATE_HOME", str(state))
    monkeypatch.setenv("NUMPILOT_CONFIG_HOME", str(config))
    monkeypatch.setenv("NUMPILOT_BOOTSTRAP_RECEIPT", str(receipt))
    monkeypatch.setenv("NUMPILOT_ASNUMPY_CACHE", str(asnumpy_base))
    monkeypatch.setenv("NUMPILOT_CANNBOT_CACHE", str(cannbot_base))

    assert _resolved_user_paths() == {
        "state": state,
        "llm_config": config / "llm.json",
        "receipt": receipt,
        "asnumpy": asnumpy_base / ASNUMPY_COMMIT,
        "cannbot": cannbot_base / CANNBOT_COMMIT,
    }


def test_numpilot_home_overrides_xdg_and_transient_home(monkeypatch, tmp_path):
    _clear_path_environment(monkeypatch)
    root = tmp_path / "persistent"
    monkeypatch.setenv("HOME", str(tmp_path / "transient-home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("NUMPILOT_HOME", str(root))

    assert _resolved_user_paths() == {
        "state": root / ".numpilot-state",
        "llm_config": root / ".numpilot-config" / "llm.json",
        "receipt": root / ".numpilot-state" / "bootstrap" / "receipt.json",
        "asnumpy": (
            root / ".numpilot-cache" / "numpilot" / "asnumpy" / ASNUMPY_COMMIT
        ),
        "cannbot": (
            root
            / ".numpilot-cache"
            / "numpilot"
            / "cannbot-skills"
            / CANNBOT_COMMIT
        ),
    }


def test_xdg_paths_override_home_when_persistent_root_is_unset(monkeypatch, tmp_path):
    _clear_path_environment(monkeypatch)
    state = tmp_path / "xdg-state"
    config = tmp_path / "xdg-config"
    cache = tmp_path / "xdg-cache"
    monkeypatch.setenv("HOME", str(tmp_path / "transient-home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))

    assert _resolved_user_paths() == {
        "state": state / "numpilot",
        "llm_config": config / "numpilot" / "llm.json",
        "receipt": state / "numpilot" / "bootstrap" / "receipt.json",
        "asnumpy": cache / "numpilot" / "asnumpy" / ASNUMPY_COMMIT,
        "cannbot": cache / "numpilot" / "cannbot-skills" / CANNBOT_COMMIT,
    }


def test_home_paths_are_the_final_fallback(monkeypatch, tmp_path):
    _clear_path_environment(monkeypatch)
    home = tmp_path / "persistent-home"
    monkeypatch.setenv("HOME", str(home))

    assert _resolved_user_paths() == {
        "state": home / ".local" / "state" / "numpilot",
        "llm_config": home / ".config" / "numpilot" / "llm.json",
        "receipt": home / ".local" / "state" / "numpilot" / "bootstrap" / "receipt.json",
        "asnumpy": home / ".cache" / "numpilot" / "asnumpy" / ASNUMPY_COMMIT,
        "cannbot": home / ".cache" / "numpilot" / "cannbot-skills" / CANNBOT_COMMIT,
    }


def test_numpilot_cache_home_is_more_specific_than_persistent_root(
    monkeypatch, tmp_path
):
    _clear_path_environment(monkeypatch)
    cache = tmp_path / "specific-numpilot-cache"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("NUMPILOT_HOME", str(tmp_path / "persistent"))
    monkeypatch.setenv("NUMPILOT_CACHE_HOME", str(cache))

    assert default_asnumpy_checkout() == cache / "asnumpy" / ASNUMPY_COMMIT
    assert default_skill_checkout() == cache / "cannbot-skills" / CANNBOT_COMMIT


def test_exact_cannbot_checkout_overrides_cache_base_for_fresh_shells(
    monkeypatch, tmp_path
):
    _clear_path_environment(monkeypatch)
    exact = tmp_path / "manual-cannbot-checkout"
    monkeypatch.setenv("NUMPILOT_CANNBOT_SKILLS", str(exact))
    monkeypatch.setenv("NUMPILOT_CANNBOT_CACHE", str(tmp_path / "unused-base"))

    assert default_skill_checkout() == exact

    _verify_default_cannbot_path(
        monkeypatch,
        {
            "HOME": str(tmp_path / "transient-home"),
            "NUMPILOT_CANNBOT_SKILLS": str(exact),
            "NUMPILOT_CANNBOT_CACHE": str(tmp_path / "unused-base"),
        },
        exact,
    )


def test_llm_config_file_override_remains_more_specific_than_all_homes(
    monkeypatch, tmp_path
):
    _clear_path_environment(monkeypatch)
    exact = tmp_path / "exact" / "model-config.json"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("NUMPILOT_HOME", str(tmp_path / "persistent"))
    monkeypatch.setenv("NUMPILOT_CONFIG_HOME", str(tmp_path / "specific-config"))
    monkeypatch.setenv("NUMPILOT_LLM_CONFIG_FILE", str(exact))

    assert llm_config_path() == exact


class _OfflineBundle:
    def manifest(self):
        return {
            "repository": bootstrap.CANNBOT_REPOSITORY,
            "commit": bootstrap.CANNBOT_COMMIT,
            "license": bootstrap.CANNBOT_LICENSE,
            "bundle_sha256": "3" * 64,
            "files": {
                item["path"]: "4" * 64
                for item in bootstrap.CANNBOT_ASCENDC_SKILL_CATALOG
            },
        }


def _verify_default_cannbot_path(monkeypatch, environment, expected):
    expected.mkdir(parents=True)
    observed = []
    monkeypatch.setattr(
        bootstrap,
        "verify_asnumpy_installation",
        lambda **_kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(
        bootstrap,
        "fetch_pinned_skill",
        lambda *_args, **_kwargs: pytest.fail("offline verify attempted a fetch"),
    )

    result = bootstrap.verify_bootstrap_installation(
        use_cannbot=True,
        accept_cannbot_license=True,
        environment=environment,
        cannbot_loader=lambda path: observed.append(path) or _OfflineBundle(),
    )

    assert observed == [expected]
    assert result["network_requested_by_numpilot"] is False
    assert result["cannbot"]["status"] == "passed"


def test_bootstrap_treats_cannbot_environment_override_as_a_cache_base(
    monkeypatch, tmp_path
):
    base = tmp_path / "cannbot-base"
    expected = base / CANNBOT_COMMIT
    _verify_default_cannbot_path(
        monkeypatch,
        {
            "HOME": str(tmp_path / "transient-home"),
            "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
            "NUMPILOT_HOME": str(tmp_path / "persistent"),
            "NUMPILOT_CANNBOT_CACHE": str(base),
        },
        expected,
    )


def test_bootstrap_uses_persistent_root_before_xdg_for_cannbot(
    monkeypatch, tmp_path
):
    root = tmp_path / "persistent"
    expected = (
        root / ".numpilot-cache" / "numpilot" / "cannbot-skills" / CANNBOT_COMMIT
    )
    _verify_default_cannbot_path(
        monkeypatch,
        {
            "HOME": str(tmp_path / "transient-home"),
            "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
            "NUMPILOT_HOME": str(root),
        },
        expected,
    )


def _make_fake_runtime(tmp_path):
    ascend = tmp_path / "usr" / "local" / "Ascend"
    cann_root = ascend / "cann-9.0.0"
    driver = ascend / "driver" / "lib64"
    for directory in (
        cann_root / "bin",
        cann_root / "include" / "acl",
        cann_root / "lib64" / "plugin" / "opskernel",
        cann_root / "lib64" / "cmake",
        cann_root / "opp",
        cann_root / "python" / "site-packages",
        cann_root / "devlib",
        driver,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (cann_root / "include" / "acl" / "acl.h").write_text("fixture\n")
    (cann_root / "lib64" / "libascendcl.so").write_bytes(b"fixture")

    cann_q = shlex.quote(str(cann_root))
    driver_q = shlex.quote(str(driver))
    cann_env = cann_root / "set_env.sh"
    cann_env.write_text(
        "\n".join(
            (
                "#!/usr/bin/env bash",
                f"export ASCEND_HOME_PATH={cann_q}",
                f"export ASCEND_TOOLKIT_HOME={cann_q}",
                f"export ASCEND_OPP_PATH={shlex.quote(str(cann_root / 'opp'))}",
                f"export PATH={shlex.quote(str(cann_root / 'bin'))}:${{PATH:-}}",
                "export LD_LIBRARY_PATH="
                f"{shlex.quote(str(cann_root / 'lib64'))}:"
                f"{shlex.quote(str(cann_root / 'lib64' / 'plugin' / 'opskernel'))}:"
                f"{driver_q}:{shlex.quote(str(cann_root / 'devlib'))}:"
                "${LD_LIBRARY_PATH:-}",
                "export PYTHONPATH="
                f"{shlex.quote(str(cann_root / 'python' / 'site-packages'))}:"
                "${PYTHONPATH:-}",
                "export CMAKE_PREFIX_PATH="
                f"{shlex.quote(str(cann_root / 'lib64' / 'cmake'))}:"
                "${CMAKE_PREFIX_PATH:-}",
                "",
            )
        ),
        encoding="utf-8",
    )

    venv = tmp_path / "persistent venv"
    (venv / "bin").mkdir(parents=True)
    venv_q = shlex.quote(str(venv))
    (venv / "bin" / "activate").write_text(
        "\n".join(
            (
                "if [ -n \"${_OLD_VIRTUAL_PATH:-}\" ]; then",
                "    PATH=$_OLD_VIRTUAL_PATH",
                "    unset _OLD_VIRTUAL_PATH",
                "fi",
                "_OLD_VIRTUAL_PATH=${PATH:-}",
                "export _OLD_VIRTUAL_PATH",
                f"export VIRTUAL_ENV={venv_q}",
                f"export PATH={shlex.quote(str(venv / 'bin'))}:${{PATH:-}}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return ascend, cann_root, cann_env, driver, venv


def _source_activation_twice(script, environment):
    command = r'''
. "$NUMPILOT_TEST_ACTIVATION"
. "$NUMPILOT_TEST_ACTIVATION"
"$NUMPILOT_TEST_PYTHON" - <<'PY'
import json
import os

names = (
    "NUMPILOT_HOME",
    "NUMPILOT_STATE_HOME",
    "NUMPILOT_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "NUMPILOT_CACHE_HOME",
    "NUMPILOT_BOOTSTRAP_RECEIPT",
    "NUMPILOT_ASNUMPY_CACHE",
    "NUMPILOT_CANNBOT_CACHE",
    "NUMPILOT_CANNBOT_SKILLS",
    "VIRTUAL_ENV",
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_OPP_PATH",
    "PATH",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "CMAKE_PREFIX_PATH",
)
print(json.dumps({name: os.environ.get(name) for name in names}, sort_keys=True))
PY
'''
    clean = dict(environment)
    clean["NUMPILOT_TEST_ACTIVATION"] = str(script)
    clean["NUMPILOT_TEST_PYTHON"] = sys.executable
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=clean,
    )


def _colon_parts(payload, name):
    return [item for item in (payload[name] or "").split(os.pathsep) if item]


def test_generated_activation_is_secret_free_and_idempotent_in_a_fresh_home(
    monkeypatch, tmp_path
):
    from numpilot.activation import write_activation_script

    _clear_path_environment(monkeypatch)
    root = tmp_path / "persistent root"
    cache = root / ".numpilot-cache"
    state = root / ".numpilot-state"
    asnumpy_base = cache / "numpilot" / "asnumpy"
    cannbot_base = cache / "numpilot" / "cannbot-skills"
    receipt = state / "bootstrap" / "receipt.json"
    activation = root / "activate-numpilot.sh"
    ascend, cann_root, cann_env, driver, venv = _make_fake_runtime(tmp_path)

    canaries = {
        "NUMPILOT_LLM_AUTH_TOKEN": "TEST_AUTH_TOKEN_CANARY",
        "NUMPILOT_LLM_BASE_URL": "https://base-url-canary.invalid/v1",
        "HTTPS_PROXY": "http://proxy-canary.invalid:9999",
        "PIP_INDEX_URL": "https://pip-index-canary.invalid/simple",
    }
    for name, value in canaries.items():
        monkeypatch.setenv(name, value)

    written = write_activation_script(
        destination=activation,
        persistent_root=root,
        venv_dir=venv,
        cann_env_file=cann_env,
        asnumpy_cache=asnumpy_base,
        cannbot_cache=cannbot_base,
        receipt_path=receipt,
    )

    assert Path(written) == activation
    assert stat.S_IMODE(activation.stat().st_mode) == 0o600
    source = activation.read_text(encoding="utf-8")
    for value in canaries.values():
        assert value not in source
    for name in ("AUTH_TOKEN", "BASE_URL", "HTTPS_PROXY", "PIP_INDEX_URL"):
        assert name not in source

    stale_cann = ascend / "cann-8.5.0"
    cross_prefix_stale = tmp_path / "other-prefix" / "Ascend" / "cann-7.0.0"
    stale_latest = ascend / "ascend-toolkit" / "latest"
    custom_op = tmp_path / "custom-opp" / "op_api" / "lib"
    unrelated = tmp_path / "unrelated" / "lib"
    unrelated_devlib = tmp_path / "unrelated" / "devlib"
    atb = ascend / "nnal" / "atb" / "latest" / "lib"
    fresh_home = tmp_path / "empty-home"
    for directory in (
        custom_op,
        unrelated,
        unrelated_devlib,
        atb,
        fresh_home,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    result = _source_activation_twice(
        activation,
        {
            "HOME": str(fresh_home),
            "PATH": os.pathsep.join(
                (str(stale_cann / "bin"), "/usr/bin", "/bin")
            ),
            "LD_LIBRARY_PATH": os.pathsep.join(
                (
                    str(custom_op),
                    str(stale_cann / "lib64"),
                    str(cross_prefix_stale / "lib64"),
                    str(stale_latest / "lib64"),
                    str(stale_cann / "devlib"),
                    str(cann_root / "devlib"),
                    str(atb),
                    str(unrelated),
                    str(unrelated_devlib),
                    str(driver) + "/",
                    str(custom_op) + "///",
                )
            ),
            "PYTHONPATH": os.pathsep.join(
                (str(stale_latest / "python" / "site-packages"), str(unrelated))
            ),
            "CMAKE_PREFIX_PATH": os.pathsep.join(
                (str(stale_cann / "lib64" / "cmake"), str(unrelated))
            ),
            "NUMPILOT_HOME": str(tmp_path / "stale-root"),
            "NUMPILOT_STATE_HOME": str(tmp_path / "stale-state"),
            "NUMPILOT_CONFIG_HOME": str(tmp_path / "stale-config"),
            "XDG_CACHE_HOME": str(tmp_path / "stale-cache"),
            "NUMPILOT_BOOTSTRAP_RECEIPT": str(tmp_path / "stale-receipt"),
        },
    )

    assert result.returncode == 0, result.stderr
    for value in canaries.values():
        assert value not in result.stdout
        assert value not in result.stderr
    payload = json.loads(result.stdout)
    assert payload["NUMPILOT_HOME"] == str(root)
    assert payload["NUMPILOT_STATE_HOME"] == str(state)
    assert payload["NUMPILOT_CONFIG_HOME"] == str(root / ".numpilot-config")
    assert payload["XDG_CACHE_HOME"] == str(cache)
    assert payload["NUMPILOT_CACHE_HOME"] == str(cache / "numpilot")
    assert payload["NUMPILOT_BOOTSTRAP_RECEIPT"] == str(receipt)
    assert payload["NUMPILOT_ASNUMPY_CACHE"] == str(asnumpy_base)
    assert payload["NUMPILOT_CANNBOT_CACHE"] == str(cannbot_base)
    assert payload["NUMPILOT_CANNBOT_SKILLS"] == str(
        cannbot_base / CANNBOT_COMMIT
    )
    assert payload["VIRTUAL_ENV"] == str(venv)
    assert payload["ASCEND_HOME_PATH"] == str(cann_root)
    assert payload["ASCEND_TOOLKIT_HOME"] == str(cann_root)
    assert payload["ASCEND_OPP_PATH"] == str(cann_root / "opp")

    path = _colon_parts(payload, "PATH")
    libraries = _colon_parts(payload, "LD_LIBRARY_PATH")
    python_path = _colon_parts(payload, "PYTHONPATH")
    cmake_path = _colon_parts(payload, "CMAKE_PREFIX_PATH")
    assert path[0] == str(venv / "bin")
    assert path.count(str(venv / "bin")) == 1
    assert path.count(str(cann_root / "bin")) == 1
    assert str(stale_cann / "bin") not in path
    assert libraries.count(str(cann_root / "lib64")) == 1
    assert libraries.count(str(cann_root / "lib64" / "plugin" / "opskernel")) == 1
    assert libraries.count(str(driver)) == 1
    assert str(cann_root / "devlib") not in libraries
    assert str(stale_cann / "devlib") not in libraries
    assert str(stale_cann / "lib64") not in libraries
    assert str(cross_prefix_stale / "lib64") not in libraries
    assert str(stale_latest / "lib64") not in libraries
    assert libraries.count(str(custom_op)) == 1
    assert str(custom_op) + "///" not in libraries
    assert str(driver) + "/" not in libraries
    assert libraries.count(str(atb)) == 1
    assert libraries.count(str(unrelated)) == 1
    assert libraries.count(str(unrelated_devlib)) == 1
    assert python_path.count(str(cann_root / "python" / "site-packages")) == 1
    assert str(stale_latest / "python" / "site-packages") not in python_path
    assert python_path.count(str(unrelated)) == 1
    assert cmake_path.count(str(cann_root / "lib64" / "cmake")) == 1
    assert str(stale_cann / "lib64" / "cmake") not in cmake_path
    assert cmake_path.count(str(unrelated)) == 1


def test_activation_without_persistent_root_freezes_standard_home_paths(
    monkeypatch, tmp_path
):
    from numpilot.activation import write_activation_script

    _clear_path_environment(monkeypatch)
    home = tmp_path / "persistent-home"
    monkeypatch.setenv("HOME", str(home))
    ascend, _cann_root, cann_env, _driver, venv = _make_fake_runtime(tmp_path)
    state = home / ".local" / "state" / "numpilot"
    cache = home / ".cache"
    asnumpy_base = cache / "numpilot" / "asnumpy"
    cannbot_base = cache / "numpilot" / "cannbot-skills"
    receipt = state / "bootstrap" / "receipt.json"
    activation = state / "activate-numpilot.sh"

    write_activation_script(
        destination=activation,
        persistent_root=None,
        venv_dir=venv,
        cann_env_file=cann_env,
        asnumpy_cache=asnumpy_base,
        cannbot_cache=cannbot_base,
        receipt_path=receipt,
    )

    source = activation.read_text(encoding="utf-8")
    assert "export NUMPILOT_HOME=" not in source
    fresh_home = tmp_path / "different-empty-home"
    fresh_home.mkdir()
    result = _source_activation_twice(
        activation,
        {
            "HOME": str(fresh_home),
            "PATH": "/usr/bin:/bin",
            "LD_LIBRARY_PATH": str(ascend / "cann-8.5.0" / "devlib"),
        },
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["NUMPILOT_HOME"] is None
    assert payload["NUMPILOT_STATE_HOME"] == str(state)
    assert payload["NUMPILOT_CONFIG_HOME"] == str(home / ".config" / "numpilot")
    assert payload["XDG_CACHE_HOME"] == str(cache)
    assert payload["NUMPILOT_CACHE_HOME"] == str(cache / "numpilot")
    assert payload["NUMPILOT_BOOTSTRAP_RECEIPT"] == str(receipt)
    assert payload["NUMPILOT_ASNUMPY_CACHE"] == str(asnumpy_base)
    assert payload["NUMPILOT_CANNBOT_CACHE"] == str(cannbot_base)
