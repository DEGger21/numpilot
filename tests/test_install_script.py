from pathlib import Path
import stat
import subprocess


SCRIPT = Path(__file__).parents[1] / "install.sh"


def _run(*arguments):
    return subprocess.run(
        [str(SCRIPT), *arguments],
        check=False,
        capture_output=True,
        text=True,
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


def test_source_installer_uses_explicit_bootstrap_plan_apply_verify_sequence():
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"$VENV_PYTHON" -m pip "${PIP_ARGS[@]}" -e "$SCRIPT_DIR[bootstrap]"' in source
    assert '"$VENV_PYTHON" -m pip "${PIP_ARGS[@]}" -e "$SCRIPT_DIR"' in source
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
    pip_install = source.index('"$VENV_PYTHON" -m pip')

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
    activation = source.index('"$VENV_PYTHON" -m numpilot.activation')
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
