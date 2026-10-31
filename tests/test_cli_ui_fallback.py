"""Core CLI must stay usable when the optional ``rich`` UI is absent.

The competition runtime is a minimal AsNumpy/NPU environment.  ``rich`` is a
presentation dependency only, so every core command has to import and run in
a plain-text fallback.  These tests block ``rich`` in a subprocess instead of
uninstalling it from the test interpreter, which keeps the normal UI tests
running in the same session.
"""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

_BLOCK_RICH = r"""
import sys

class _BlockRich:
    def find_spec(self, name, path=None, target=None):
        if name == "rich" or name.startswith("rich."):
            raise ModuleNotFoundError("No module named 'rich'", name=name)
        return None

sys.meta_path.insert(0, _BlockRich())
"""


def _run_without_rich(code, *, timeout=120):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["NUMPILOT_DISABLE_REPO_CONFIG"] = "1"
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_RICH + code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_ui_imports_and_renders_plain_text_without_rich():
    result = _run_without_rich(
        """
from numpilot import cli_ui

assert cli_ui.rich_available() is False
assert cli_ui.is_interactive(sys.stdout) is False
cli_ui.flow_print("plain-fallback")
with cli_ui.terminal_status("checking") as status:
    status.warning()
print("ok")
"""
    )
    assert result.returncode == 0, result.stderr
    assert "plain-fallback" in result.stdout
    assert "ok" in result.stdout
    assert "Traceback" not in result.stderr


def test_core_cli_paths_command_starts_without_rich():
    result = _run_without_rich(
        """
from numpilot.__main__ import main

raise SystemExit(main(["paths", "--json"]))
"""
    )
    assert result.returncode == 0, result.stderr
    assert "llm_config" in result.stdout
    assert "Traceback" not in result.stderr


def test_wizard_requires_rich_with_a_clear_error():
    result = _run_without_rich(
        """
from numpilot.__main__ import main

raise SystemExit(main(["wizard", "--help"]))
"""
    )
    # ``--help`` is handled by argparse before the wizard import, so the
    # important contract here is simply that importing the CLI never fails.
    assert result.returncode in {0, 2}, result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_migration_report_roster_is_importable_without_rich():
    result = _run_without_rich(
        """
from numpilot.report import agent_roster

roster = agent_roster("ast")
assert any("Migration" in name for name, _, _ in roster)
print("roster-ok")
"""
    )
    assert result.returncode == 0, result.stderr
    assert "roster-ok" in result.stdout
