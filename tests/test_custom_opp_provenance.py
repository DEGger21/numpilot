from pathlib import Path
import json

import pytest

from numpilot.custom_opp_provenance import (
    CustomOppProvenanceError,
    inspect_custom_opp,
)


def _vendor(tmp_path: Path) -> Path:
    root = tmp_path / "softplus_custom"
    (root / "op_api/include").mkdir(parents=True)
    (root / "op_api/lib").mkdir(parents=True)
    (root / "version.info").write_text(
        "custom_opp_compiler_version=9.2.0-beta.2\n",
        encoding="utf-8",
    )
    (root / "op_api/include/aclnn_softplus.h").write_text(
        "aclError aclnnSoftplusGetWorkspaceSize();\n"
        "aclError aclnnSoftplus();\n",
        encoding="utf-8",
    )
    (root / "op_api/lib/libcust_opapi.so").write_bytes(b"test-library")
    return root


def _inspect(root: Path, **overrides):
    values = {
        "operator": "softplus",
        "get_workspace_size": "aclnnSoftplusGetWorkspaceSize",
        "execute": "aclnnSoftplus",
    }
    values.update(overrides)
    return inspect_custom_opp(root, **values)


def test_inspect_custom_opp_returns_hash_bound_receipt(tmp_path):
    receipt = _inspect(
        _vendor(tmp_path),
        expected_compiler_version="9.2.0-beta.2",
    )
    data = receipt.to_dict()
    assert data["schema"] == "numpilot.custom-opp-provenance"
    assert data["aclnn"]["call_style"] == "two-stage"
    assert data["library"]["size"] == len(b"test-library")
    assert len(data["receipt_sha256"]) == 64
    assert json.loads(json.dumps(data)) == data


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operator", "../softplus"),
        ("get_workspace_size", "aclnnSoftplus"),
        ("execute", "aclnnSoftplusGetWorkspaceSize"),
    ],
)
def test_inspect_custom_opp_rejects_invalid_contract(tmp_path, field, value):
    with pytest.raises(CustomOppProvenanceError):
        _inspect(_vendor(tmp_path), **{field: value})


def test_inspect_custom_opp_rejects_version_mismatch(tmp_path):
    with pytest.raises(CustomOppProvenanceError, match="compiler version"):
        _inspect(_vendor(tmp_path), expected_compiler_version="9.3.0")


def test_inspect_custom_opp_rejects_missing_header_symbol(tmp_path):
    root = _vendor(tmp_path)
    (root / "op_api/include/aclnn_softplus.h").write_text(
        "aclError aclnnSoftplusGetWorkspaceSize();\n", encoding="utf-8"
    )
    with pytest.raises(CustomOppProvenanceError, match="does not declare"):
        _inspect(root)


def test_inspect_custom_opp_rejects_symlinked_library(tmp_path):
    root = _vendor(tmp_path)
    target = root / "op_api/lib/libcust_opapi.so"
    target.unlink()
    target.symlink_to(root / "version.info")
    with pytest.raises(CustomOppProvenanceError):
        _inspect(root)
