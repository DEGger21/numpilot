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

import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import stat
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "install.sh"


def _run(*arguments, env=None):
    return subprocess.run(
        [str(SCRIPT), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_source_installer_is_executable_and_has_valid_shell_syntax():
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & stat.S_IXUSR
    checked = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr


def test_source_installer_help_describes_one_click_and_manual_checkouts():
    result = _run("--help")

    assert result.returncode == 0
    assert "--allow-network" in result.stdout
    assert "--python" in result.stdout
    assert "--pip-index-url" in result.stdout
    assert "--asnumpy-source" in result.stdout
    assert "--asnumpy-component" in result.stdout
    assert "--cannbot-source" in result.stdout
    assert "--persistent-root" in result.stdout
    assert "activate-numpilot.sh" in result.stdout
    assert "never runs sudo" in result.stdout


def test_source_installer_fails_before_writes_without_network_consent(tmp_path):
    venv = tmp_path / "must-not-exist"
    result = _run("--venv", str(venv))

    assert result.returncode == 2
    assert "refusing implicit network access" in result.stderr
    assert not venv.exists()


def test_source_installer_fails_before_writes_without_cannbot_license(tmp_path):
    venv = tmp_path / "must-not-exist"
    result = _run(
        "--allow-network",
        "--with-cannbot",
        "--venv",
        str(venv),
    )

    assert result.returncode == 2
    assert "requires --accept-cannbot-license" in result.stderr
    assert not venv.exists()


def test_source_installer_requires_existing_explicit_cannbot_checkout(tmp_path):
    venv = tmp_path / "must-not-exist"
    missing_checkout = tmp_path / "missing-cannbot-checkout"
    result = _run(
        "--allow-network",
        "--cannbot-source",
        str(missing_checkout),
        "--accept-cannbot-license",
        "--venv",
        str(venv),
    )

    assert result.returncode == 2
    assert "must name an existing pinned checkout" in result.stderr
    assert not venv.exists()


def test_source_installer_uses_explicit_bootstrap_plan_apply_verify_sequence():
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"$VENV_PYTHON" -I -m pip "${PIP_ARGS[@]}" -e "$SCRIPT_DIR[bootstrap]"' in source
    assert '"$VENV_PYTHON" -I -m pip "${PIP_ARGS[@]}" -e "$SCRIPT_DIR"' in source
    assert "heavier build extras" in source
    assert 'export PATH="$VENV_DIR/bin:$PATH"' in source
    assert 'bootstrap --plan "${BOOTSTRAP_ARGS[@]}"' in source
    assert 'export NUMPILOT_BOOTSTRAP_RECEIPT=$RECEIPT_PATH' in source
    assert 'APPLY_ARGS=(--apply "${BOOTSTRAP_ARGS[@]}" --receipt "$RECEIPT_PATH")' in source
    assert 'VERIFY_ARGS=(--verify)' in source
    assert "doctor --backend npu" in source
    assert "choose exactly one of --asnumpy-component or maintainer-only --asnumpy-source" in source
    assert "sudo " not in "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


def test_source_installer_validates_receipt_before_bootstrap():
    source = SCRIPT.read_text(encoding="utf-8")
    validation = source.index("receipt path failed the private-owner safety policy")
    plan = source.index('bootstrap --plan "${BOOTSTRAP_ARGS[@]}"')

    assert validation < plan
    assert "os.path.abspath(os.path.expanduser(raw_path))" in source
    assert "parent.lstat()" in source
    assert "receipt.lstat()" in source
    assert "stat.S_ISLNK(parent_status.st_mode)" in source
    assert "stat.S_ISLNK(receipt_status.st_mode)" in source
    assert "parent_status.st_uid != current_uid" in source
    assert "receipt_status.st_uid != current_uid" in source
    assert "parent_status.st_mode & 0o022" in source
    assert "except FileNotFoundError:\n    pass" in source


def test_source_installer_prepares_private_default_receipt_parent():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "parent.mkdir(parents=True, mode=0o700)" in source
    assert "os.chmod(parent, 0o700)" in source
    assert "stat.S_IMODE(parent_status.st_mode) != 0o700" in source
    assert "custom receipt parent must already exist" in source


def test_source_installer_rejects_system_or_fake_venv_paths_before_pip():
    source = SCRIPT.read_text(encoding="utf-8")
    path_policy = source.index("venv path failed the private non-system safety policy")
    interpreter_policy = source.index("--venv must name a genuine isolated virtual environment")
    pip_install = source.index('"$VENV_PYTHON" -I -m pip')

    assert path_policy < interpreter_policy < pip_install
    assert 'Path("/usr")' in source
    assert "venv path must not traverse a symlink" in source
    assert "venv parent must be user-owned and not group/world writable" in source
    assert "sys.prefix == sys.base_prefix" in source
    assert 'configuration = expected / "pyvenv.cfg"' in source


def test_source_installer_prints_shell_quoted_activation_and_receipt_paths():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "activate: source %q" in source
    assert "receipt: %q" in source
    assert "later shells: export NUMPILOT_BOOTSTRAP_RECEIPT" not in source


def test_source_installer_applies_persistent_root_before_bootstrap_and_writes_activation():
    source = SCRIPT.read_text(encoding="utf-8")

    parse = source.index("--persistent-root)")
    root_export = source.index("export NUMPILOT_HOME=$PERSISTENT_ROOT")
    bootstrap = source.index('bootstrap --plan "${BOOTSTRAP_ARGS[@]}"')
    activation = source.index('"$VENV_PYTHON" -I -m numpilot.activation')
    activation_source = source.index('. "$ACTIVATION_PATH"')
    verify = source.index('bootstrap "${VERIFY_ARGS[@]}"')
    doctor = source.index('doctor --backend npu')

    assert (
        parse
        < root_export
        < activation
        < activation_source
        < bootstrap
        < verify
        < doctor
    )
    assert 'export NUMPILOT_STATE_HOME="$PERSISTENT_ROOT/.numpilot-state"' in source
    assert 'export NUMPILOT_CONFIG_HOME="$PERSISTENT_ROOT/.numpilot-config"' in source
    assert 'export XDG_CACHE_HOME="$PERSISTENT_ROOT/.numpilot-cache"' in source
    assert 'ACTIVATION_PATH="$PERSISTENT_ROOT/activate-numpilot.sh"' in source
    assert "activate: source %q" in source
    assert ".bashrc" not in source
    assert ".codex-env.sh" not in source


def test_source_installer_routes_preflight_and_venv_operations_through_selected_python():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'PYTHON_COMMAND="python3"' in source
    assert 'PYTHON_COMMAND=$(command -v -- "$PYTHON_REQUEST")' in source
    assert '"$PYTHON_COMMAND" -I -B -' in source
    assert 'PERSISTENT_ROOT=$("$PYTHON_COMMAND" -I - "$PERSISTENT_ROOT"' in source
    assert 'VENV_DIR=$("$PYTHON_COMMAND" -I - "$VENV_DIR" "$SCRIPT_DIR"' in source
    assert '"$PYTHON_COMMAND" -I -m venv "$VENV_DIR"' in source
    assert "python3 - <<'PY'" not in source
    assert 'python3 -m venv "$VENV_DIR"' not in source


def test_all_venv_module_entry_points_ignore_external_pythonpath():
    source = SCRIPT.read_text(encoding="utf-8")
    module_invocations = [
        line.strip()
        for line in source.splitlines()
        if '"$VENV_PYTHON"' in line and " -m " in line
    ]

    assert module_invocations
    assert all('"$VENV_PYTHON" -I -m ' in line for line in module_invocations)
    assert '"$VENV_PYTHON" -m ' not in source


def test_source_installer_uses_explicit_python_for_component_preflight(tmp_path):
    log = tmp_path / "python-invocations.log"
    wrapper = tmp_path / "selected-python"
    wrapper.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "$NUMPILOT_TEST_PYTHON_LOG"\n'
        'exec "$NUMPILOT_TEST_REAL_PYTHON" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    missing_component = tmp_path / "missing-component"
    venv = tmp_path / "must-not-exist"
    environment = {
        **os.environ,
        "NUMPILOT_TEST_PYTHON_LOG": str(log),
        "NUMPILOT_TEST_REAL_PYTHON": sys.executable,
    }

    result = _run(
        "--allow-network",
        "--python",
        str(wrapper),
        "--asnumpy-component",
        str(missing_component),
        "--venv",
        str(venv),
        env=environment,
    )

    assert result.returncode == 2
    assert "component preflight failed" in result.stderr
    assert log.read_text(encoding="utf-8").startswith("-I -B - ")
    assert not venv.exists()


@pytest.mark.parametrize(
    ("runtime_key", "incompatible_value", "error"),
    (
        ("python_implementation", "PyPy", "must declare CPython"),
        ("python_major_minor", "0.0", "requires Python 0.0"),
        ("machine", "unsupported_machine", "requires machine unsupported_machine"),
    ),
)
def test_frozen_component_runtime_mismatch_fails_before_venv_or_pip(
    tmp_path,
    runtime_key,
    incompatible_value,
    error,
):
    component = tmp_path / "component"
    shutil.copytree(SCRIPT.parent / "components" / "asnumpy", component)
    manifest_path = component / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"][runtime_key] = incompatible_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    venv = tmp_path / "must-not-exist"

    result = _run(
        "--allow-network",
        "--python",
        sys.executable,
        "--asnumpy-component",
        str(component),
        "--venv",
        str(venv),
    )

    assert result.returncode == 2
    assert error in result.stderr
    assert not venv.exists()


def test_frozen_component_pip_ignores_external_pythonpath_and_uses_exact_pins(
    tmp_path,
):
    manifest = json.loads(
        (SCRIPT.parent / "components" / "asnumpy" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    runtime = manifest["runtime"]
    assert runtime["python_implementation"] == platform.python_implementation()
    assert runtime["python_major_minor"] == "%d.%d" % sys.version_info[:2]
    assert runtime["machine"] == platform.machine()

    cann_root = tmp_path / "fake-cann"
    opp = cann_root / "opp"
    (cann_root / "include" / "acl").mkdir(parents=True)
    (cann_root / "lib64").mkdir()
    opp.mkdir()
    (cann_root / "include" / "acl" / "acl.h").write_bytes(b"")
    (cann_root / "lib64" / "libascendcl.so").write_bytes(b"")
    cann_env = cann_root / "set_env.sh"
    cann_env.write_text(
        "\n".join(
            (
                "export ASCEND_TOOLKIT_HOME=%s" % shlex.quote(str(cann_root)),
                "export ASCEND_HOME_PATH=%s" % shlex.quote(str(cann_root)),
                "export ASCEND_OPP_PATH=%s" % shlex.quote(str(opp)),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    venv = tmp_path / "venv"
    created = subprocess.run(
        [sys.executable, "-I", "-m", "venv", "--without-pip", str(venv)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert created.returncode == 0, created.stderr

    venv_python = venv / "bin" / "python"
    purelib_result = subprocess.run(
        [
            str(venv_python),
            "-I",
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert purelib_result.returncode == 0, purelib_result.stderr
    controlled_pip = Path(purelib_result.stdout.strip()) / "pip"
    controlled_pip.mkdir(parents=True)
    (controlled_pip / "__init__.py").write_text("", encoding="utf-8")
    (controlled_pip / "__main__.py").write_text(
        "from pathlib import Path\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "Path(os.environ['NUMPILOT_TEST_CONTROLLED_PIP_LOG']).write_text(\n"
        "    json.dumps(sys.argv[1:]), encoding='utf-8'\n"
        ")\n"
        "raise SystemExit(73)\n",
        encoding="utf-8",
    )

    external_modules = tmp_path / "external-modules"
    external_pip = external_modules / "pip"
    external_pip.mkdir(parents=True)
    (external_pip / "__init__.py").write_text("", encoding="utf-8")
    (external_pip / "__main__.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['NUMPILOT_TEST_EXTERNAL_PIP_LOG']).write_text(\n"
        "    'executed', encoding='utf-8'\n"
        ")\n"
        "raise SystemExit(72)\n",
        encoding="utf-8",
    )
    controlled_pip_log = tmp_path / "controlled-pip-arguments.json"
    external_pip_log = tmp_path / "external-pip-executed"
    environment = {
        **os.environ,
        "PYTHONPATH": str(external_modules),
        "NUMPILOT_TEST_CONTROLLED_PIP_LOG": str(controlled_pip_log),
        "NUMPILOT_TEST_EXTERNAL_PIP_LOG": str(external_pip_log),
    }

    result = _run(
        "--allow-network",
        "--python",
        sys.executable,
        "--cann-env",
        str(cann_env),
        "--venv",
        str(venv),
        env=environment,
    )

    assert result.returncode == 73, result.stderr
    assert not external_pip_log.exists()
    arguments = json.loads(controlled_pip_log.read_text(encoding="utf-8"))
    assert "numpy==%s" % runtime["numpy_version"] in arguments
    assert "loguru==%s" % runtime["loguru_version"] in arguments
    assert [item for item in arguments if item.startswith("numpy==")] == [
        "numpy==%s" % runtime["numpy_version"]
    ]
    assert [item for item in arguments if item.startswith("loguru==")] == [
        "loguru==%s" % runtime["loguru_version"]
    ]


def test_frozen_component_preflight_and_pins_precede_pip():
    source = SCRIPT.read_text(encoding="utf-8")

    preflight = source.index("component_module.load_component_manifest")
    cann_discovery = source.index('if [ -z "$CANN_ENV_FILE" ]')
    venv_creation = source.index('"$PYTHON_COMMAND" -I -m venv "$VENV_DIR"')
    venv_runtime_check = source.index(
        'fail "--venv interpreter does not satisfy the frozen AsNumpy component runtime"'
    )
    pip_install = source.index('"$VENV_PYTHON" -I -m pip')

    assert preflight < cann_discovery < venv_creation < venv_runtime_check < pip_install
    assert '"numpy==$COMPONENT_NUMPY_VERSION"' in source
    assert '"loguru==$COMPONENT_LOGURU_VERSION"' in source
