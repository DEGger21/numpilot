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
    assert "--cannbot-source" in result.stdout
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
    assert 'export PATH="$VENV_DIR/bin:$PATH"' in source
    assert 'bootstrap --plan "${BOOTSTRAP_ARGS[@]}"' in source
    assert 'export NUMPILOT_BOOTSTRAP_RECEIPT=$RECEIPT_PATH' in source
    assert 'APPLY_ARGS=(--apply "${BOOTSTRAP_ARGS[@]}" --receipt "$RECEIPT_PATH")' in source
    assert 'VERIFY_ARGS=(--verify)' in source
    assert "doctor --backend npu" in source
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


def test_source_installer_prints_only_a_shell_quoted_receipt_export():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "NUMPILOT_BOOTSTRAP_RECEIPT=%q" in source
    assert (
        "custom receipt runtime: export "
        "NUMPILOT_BOOTSTRAP_RECEIPT=$RECEIPT_PATH"
    ) not in source
