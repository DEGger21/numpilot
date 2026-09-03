"""Materialize portable recovery receipts for tests.

The checked-in JSON under ``tests/fixtures`` is deliberately a template: it
contains symbolic path tokens instead of a developer's checkout or toolkit
directories.  Tests that exercise the strict receipt validators need those
tokens resolved to the current checkout, so this module creates an isolated
temporary copy at import time.  The production validators remain strict and
never learn about the template tokens.
"""

from __future__ import annotations

import atexit
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ops-nn-recovery"
_SOURCE = (_REPO_ROOT / "examples" / "ops_nn_matmul_inference.py").resolve()
_MATERIALIZED: Path | None = None


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_tokens(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_tokens(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_tokens(item) for item in value]
    if value == "__NUMPILOT_SOURCE__":
        return str(_SOURCE)
    if value == "__CANN_ROOT__":
        # The fixture only records the identity; no test is allowed to read or
        # mutate this path.  Keep it absolute because the production receipt
        # contract intentionally requires absolute environment roots.
        return "/REDACTED/cann"
    if value == "__OPS_NN_ROOT__":
        return "/REDACTED/ops-nn"
    return value


def _materialize_json(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value = _resolve_tokens(value)

    if path.name == "official-resolution-linkage-v2.json":
        # The official receipt binds the decision to the complete GapSpec.
        # Its source path is checkout-specific, so derive the digest from the
        # resolved template rather than carrying a stale workstation hash.
        from numpilot.gaps.models import GapSpec
        from numpilot.gaps.official_resolution import gap_spec_sha256

        request_path = path.parent / "pre-integration-gap-v2" / "gap-request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request = _resolve_tokens(request)
        specs = {
            (spec["gap_id"], spec["qualified_api"]): gap_spec_sha256(
                GapSpec.from_dict(spec)
            )
            for spec in request["gap_specs"]
        }
        for decision in value.get("decisions", []):
            key = (decision.get("gap_id"), decision.get("qualified_api"))
            if key in specs:
                decision["gap_spec_sha256"] = specs[key]
        body = {key: item for key, item in value.items() if key != "resolution_id"}
        value["resolution_id"] = _canonical_sha256(body)

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _cleanup() -> None:
    global _MATERIALIZED
    if _MATERIALIZED is None:
        return
    try:
        shutil.rmtree(_MATERIALIZED.parent)
    except OSError:
        pass
    _MATERIALIZED = None


def recovery_fixture_root() -> Path:
    """Return a temporary, checkout-bound copy of the recovery fixtures."""

    global _MATERIALIZED
    if _MATERIALIZED is not None and _MATERIALIZED.is_dir():
        return _MATERIALIZED
    parent = Path(tempfile.mkdtemp(prefix="numpilot-recovery-fixture-"))
    root = parent / "ops-nn-recovery"
    shutil.copytree(_TEMPLATE_ROOT, root)
    for path in root.rglob("*.json"):
        _materialize_json(path)
    _MATERIALIZED = root
    atexit.register(_cleanup)
    return root

