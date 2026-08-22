"""Resolve NumPilot's user-owned configuration, cache, and run paths.

Generated artifacts must not depend on the current working directory.  This
module follows the XDG base-directory convention on Unix-like systems while
keeping explicit environment overrides for CI, containers, and competition
evaluation machines.
"""
from pathlib import Path
import os
from typing import Dict


def _expanded(value: str) -> Path:
    return Path(value).expanduser()


def state_home() -> Path:
    """Return the NumPilot state root without creating it."""
    explicit = os.getenv("NUMPILOT_STATE_HOME")
    if explicit:
        return _expanded(explicit)
    xdg_state = os.getenv("XDG_STATE_HOME")
    base = _expanded(xdg_state) if xdg_state else Path.home() / ".local" / "state"
    return base / "numpilot"


def runs_dir() -> Path:
    """Return the root for generated code, benchmarks, and gap runs."""
    explicit = os.getenv("NUMPILOT_RUNS_DIR")
    return _expanded(explicit) if explicit else state_home() / "runs"


def reports_dir() -> Path:
    """Return the root for timestamped human-readable reports."""
    explicit = os.getenv("NUMPILOT_REPORTS_DIR")
    return _expanded(explicit) if explicit else state_home() / "reports"


def default_benchmark_dir() -> Path:
    return runs_dir() / "benchmark"


def default_wizard_dir() -> Path:
    return runs_dir() / "wizard"


def default_gap_dir() -> Path:
    return runs_dir() / "gaps"


def default_demo_dir() -> Path:
    return runs_dir() / "demo"


def path_summary() -> Dict[str, str]:
    """Return display-safe paths; no directory is created as a side effect."""
    from numpilot.cannbot_skills import default_skill_checkout
    from numpilot.llm_config import llm_config_path

    return {
        "state": str(state_home()),
        "runs": str(runs_dir()),
        "reports": str(reports_dir()),
        "benchmark": str(default_benchmark_dir()),
        "wizard": str(default_wizard_dir()),
        "gaps": str(default_gap_dir()),
        "demo": str(default_demo_dir()),
        "llm_config": str(llm_config_path()),
        "cannbot_cache": str(default_skill_checkout()),
    }
