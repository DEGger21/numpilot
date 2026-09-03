from __future__ import annotations

import json
from pathlib import Path

import pytest

from numpilot.asnumpy_component import (
    AsNumpyComponentError,
    load_component_manifest,
)
from numpilot.bootstrap import ASNUMPY_COMMIT, ASNUMPY_TREE, ASNUMPY_VERSION


ROOT = Path(__file__).resolve().parents[1]
LOCK_DOC = ROOT / "docs/ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md"
REFERENCE_RECEIPT = (
    ROOT / "evidence/competition-poisson-stage-20260828/bootstrap-receipt.json"
)


def test_internal_asnumpy_lock_is_bound_to_the_pinned_runtime_and_reference_receipt():
    text = LOCK_DOC.read_text(encoding="utf-8")
    receipt = json.loads(REFERENCE_RECEIPT.read_text(encoding="utf-8"))
    source = receipt["repositories"]["asnumpy"]
    wheel = receipt["wheel"]

    assert ASNUMPY_COMMIT in text
    assert ASNUMPY_TREE in text
    assert ASNUMPY_VERSION in text
    assert source["repository"] in text
    assert source["license"] == "Apache-2.0"
    assert wheel["sha256"] in text
    assert wheel["metadata_sha256"] in text
    assert wheel["record_sha256"] in text
    assert wheel["installed_files"]["module"]["sha256"] in text
    assert all(item["sha256"] in text for item in wheel["installed_files"]["native_modules"])


def test_internal_asnumpy_lock_declares_no_fetch_install_and_reviewed_upgrade_contract():
    text = LOCK_DOC.read_text(encoding="utf-8")

    for marker in (
        "components/asnumpy/",
        "--asnumpy-wheel-sha256",
        "解析 `latest`",
        "不能退回源码下载",
        "升级、回滚与安全修复流程",
        "SBOM",
        "Apache-2.0",
        "历史 wheel SHA-256",
        "当前工作树仍是 Alpha 开发快照",
    ):
        assert marker in text


def test_public_entry_points_link_to_the_component_lock():
    for relative in ("README.md", "README.zh-CN.md", "docs/README.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md" in text

    third_party = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "Ascend NPU 运行时内部组件" in third_party
    assert "不得因为文件缺失而从上游拉取源码" in third_party


@pytest.mark.parametrize("missing", ("license_file", "sbom"))
def test_frozen_component_requires_legal_attachments(tmp_path, missing):
    """A binary-only frozen lock must not pass the component parser."""

    payload = json.loads(
        (ROOT / "components/asnumpy/manifest.json").read_text(encoding="utf-8")
    )
    payload.pop(missing)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AsNumpyComponentError, match="attachment"):
        load_component_manifest(manifest)


def test_component_path_rejects_symlink_traversal_before_normalization(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "manifest.json").write_bytes(
        (ROOT / "components/asnumpy/manifest.json").read_bytes()
    )
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(AsNumpyComponentError, match="symlink"):
        load_component_manifest(link / ".." / "target" / "manifest.json")
