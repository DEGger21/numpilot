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

"""Resolve NumPilot's user-owned configuration, cache, and run paths.

Generated artifacts must not depend on the current working directory.  This
module follows the XDG base-directory convention on Unix-like systems while
keeping explicit environment overrides for CI, containers, and managed runtime
evaluation machines.
"""
from pathlib import Path
import os
from typing import Dict, Mapping, Optional


def _expanded(value: str) -> Path:
    return Path(value).expanduser()


def _environment(environment: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if environment is None else environment


def _user_home(environment: Mapping[str, str]) -> Path:
    value = environment.get("HOME")
    return _expanded(value) if value else Path.home()


def persistent_home(
    environment: Optional[Mapping[str, str]] = None,
) -> Optional[Path]:
    """Return the optional platform-persistent root without creating it."""

    values = _environment(environment)
    # ``NUMPILOT_RUNTIME_ROOT`` is an intentionally descriptive alias for
    # NUMPILOT_HOME.  It makes a separately provisioned managed runtime
    # self-documenting while preserving the historical variable's precedence.
    value = values.get("NUMPILOT_HOME") or values.get("NUMPILOT_RUNTIME_ROOT")
    return _expanded(value) if value else None


def state_home(environment: Optional[Mapping[str, str]] = None) -> Path:
    """Return the NumPilot state root without creating it."""
    values = _environment(environment)
    explicit = values.get("NUMPILOT_STATE_HOME")
    if explicit:
        return _expanded(explicit)
    root = persistent_home(values)
    if root is not None:
        return root / ".numpilot-state"
    xdg_state = values.get("XDG_STATE_HOME")
    base = (
        _expanded(xdg_state)
        if xdg_state
        else _user_home(values) / ".local" / "state"
    )
    return base / "numpilot"


def config_home(environment: Optional[Mapping[str, str]] = None) -> Path:
    """Return the NumPilot configuration root without creating it."""

    values = _environment(environment)
    explicit = values.get("NUMPILOT_CONFIG_HOME")
    if explicit:
        return _expanded(explicit)
    root = persistent_home(values)
    if root is not None:
        return root / ".numpilot-config"
    xdg_config = values.get("XDG_CONFIG_HOME")
    base = (
        _expanded(xdg_config)
        if xdg_config
        else _user_home(values) / ".config"
    )
    return base / "numpilot"


def cache_home(environment: Optional[Mapping[str, str]] = None) -> Path:
    """Return the NumPilot-owned application cache root."""

    values = _environment(environment)
    explicit = values.get("NUMPILOT_CACHE_HOME")
    if explicit:
        return _expanded(explicit)
    root = persistent_home(values)
    if root is not None:
        return root / ".numpilot-cache" / "numpilot"
    xdg_cache = values.get("XDG_CACHE_HOME")
    base = _expanded(xdg_cache) if xdg_cache else _user_home(values) / ".cache"
    return base / "numpilot"


def asnumpy_cache_base(
    environment: Optional[Mapping[str, str]] = None,
) -> Path:
    values = _environment(environment)
    explicit = values.get("NUMPILOT_ASNUMPY_CACHE")
    return _expanded(explicit) if explicit else cache_home(values) / "asnumpy"


def asnumpy_component_dir(
    environment: Optional[Mapping[str, str]] = None,
) -> Path:
    """Return the local release-adjacent AsNumpy component directory.

    The environment override is intentionally separate from the historical
    source cache.  A component directory contains manifest.json and the
    exact wheel attachment; it is never a Git checkout and is never fetched.
    """

    values = _environment(environment)
    explicit = values.get("NUMPILOT_ASNUMPY_COMPONENT_DIR")
    if explicit:
        return _expanded(explicit)
    source = Path(__file__).resolve().parents[1] / "components" / "asnumpy"
    if source.exists():
        return source
    import sys

    return Path(sys.prefix) / "share" / "numpilot" / "components" / "asnumpy"


def cannbot_cache_base(
    environment: Optional[Mapping[str, str]] = None,
) -> Path:
    values = _environment(environment)
    explicit = values.get("NUMPILOT_CANNBOT_CACHE")
    return (
        _expanded(explicit)
        if explicit
        else cache_home(values) / "cannbot-skills"
    )


def runs_dir(environment: Optional[Mapping[str, str]] = None) -> Path:
    """Return the root for generated code, benchmarks, and gap runs."""
    values = _environment(environment)
    explicit = values.get("NUMPILOT_RUNS_DIR")
    return _expanded(explicit) if explicit else state_home(values) / "runs"


def reports_dir(environment: Optional[Mapping[str, str]] = None) -> Path:
    """Return the root for timestamped human-readable reports."""
    values = _environment(environment)
    explicit = values.get("NUMPILOT_REPORTS_DIR")
    return _expanded(explicit) if explicit else state_home(values) / "reports"


def default_benchmark_dir() -> Path:
    return runs_dir() / "benchmark"


def default_wizard_dir() -> Path:
    return runs_dir() / "wizard"


def default_gap_dir() -> Path:
    return runs_dir() / "gaps"


def default_demo_dir() -> Path:
    return runs_dir() / "demo"


def default_bootstrap_receipt(
    environment: Optional[Mapping[str, str]] = None,
) -> Path:
    """Return the default supply-chain receipt path without creating it."""

    values = _environment(environment)
    explicit = values.get("NUMPILOT_BOOTSTRAP_RECEIPT")
    if explicit:
        return _expanded(explicit)
    return state_home(values) / "bootstrap" / "receipt.json"


def path_summary() -> Dict[str, str]:
    """Return display-safe paths; no directory is created as a side effect."""
    from numpilot.cannbot_skills import default_skill_checkout
    from numpilot.bootstrap import default_asnumpy_checkout
    from numpilot.llm_config import llm_config_path

    return {
        "persistent_home": str(persistent_home() or ""),
        "state": str(state_home()),
        "config": str(config_home()),
        "cache": str(cache_home()),
        "runs": str(runs_dir()),
        "reports": str(reports_dir()),
        "benchmark": str(default_benchmark_dir()),
        "wizard": str(default_wizard_dir()),
        "gaps": str(default_gap_dir()),
        "demo": str(default_demo_dir()),
        "bootstrap_receipt": str(default_bootstrap_receipt()),
        "llm_config": str(llm_config_path()),
        "asnumpy_cache_base": str(asnumpy_cache_base()),
        "asnumpy_component": str(asnumpy_component_dir()),
        "asnumpy_cache": str(default_asnumpy_checkout()),
        "cannbot_cache_base": str(cannbot_cache_base()),
        "cannbot_cache": str(default_skill_checkout()),
    }
