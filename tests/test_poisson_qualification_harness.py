"""Safety regressions for the external Poisson qualification harness."""

import importlib.util
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
