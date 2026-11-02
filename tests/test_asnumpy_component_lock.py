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

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from numpilot.asnumpy_component import (
    AsNumpyComponentError,
    component_receipt_identity,
    load_component_manifest,
    verify_component_attachments,
)
from numpilot.bootstrap import ASNUMPY_COMMIT, ASNUMPY_TREE, ASNUMPY_VERSION


ROOT = Path(__file__).resolve().parents[1]
LOCK_DOC = ROOT / "docs/ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md"
COMPONENT_MANIFEST = ROOT / "components/asnumpy/manifest.json"


def test_internal_asnumpy_lock_is_bound_to_the_bundled_component_manifest():
    text = LOCK_DOC.read_text(encoding="utf-8")
    manifest = json.loads(COMPONENT_MANIFEST.read_text(encoding="utf-8"))
    wheel = manifest["wheel"]

    assert manifest["component_id"] in text
    assert ASNUMPY_COMMIT in text
    assert ASNUMPY_TREE in text
    assert ASNUMPY_VERSION in text
    assert manifest["repository"] in text
    assert manifest["license"] == "Apache-2.0"
    assert manifest["license"] in text
    assert manifest["build"]["compiler"] in text
    assert manifest["build"]["method"] in text
    assert manifest["build"]["python"] in text
    assert wheel["filename"] in text
    assert all(tag in text for tag in wheel["tags"])
    assert wheel["sha256"] in text
    assert wheel["metadata_sha256"] in text
    assert wheel["record_sha256"] in text
    assert wheel["installed_files"]["module"]["sha256"] in text
    assert all(item["sha256"] in text for item in wheel["installed_files"]["native_modules"])
    assert manifest["license_file"]["sha256"] in text
    assert manifest["sbom"]["format"] in text
    assert manifest["sbom"]["sha256"] in text
    assert all(digest in text for digest in manifest["source_files_sha256"].values())
    assert hashlib.sha256(COMPONENT_MANIFEST.read_bytes()).hexdigest() in text


def test_internal_asnumpy_lock_omits_superseded_artifact_history():
    text = LOCK_DOC.read_text(encoding="utf-8")

    for marker in (
        "2026-09-03",
        "9822af6aa4ba49a90f31659966a02b1a628daa683ca762471b2b1b3e6a1e1155",
        "d8a107c1ff75f506c6218aead803937aae4a2afd59931f05135bf9e6a39a8d49",
        "原 wheel 字节已丢失",
        "与当前开发态的关系",
    ):
        assert marker not in text


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
        "当前版本仍处于 Alpha 阶段",
    ):
        assert marker in text


def test_product_docs_expose_the_component_lock_and_bundled_component():
    docs_index = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    assert "ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md" in docs_index

    for relative in ("README.md", "README.zh-CN.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "components/asnumpy/" in text

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


def test_component_receipt_uses_digest_of_the_same_payload_that_was_parsed(
    tmp_path, monkeypatch
):
    trusted = (ROOT / "components/asnumpy/manifest.json").read_bytes()
    replacement = trusted.replace(
        b'"sha256": "a8c47cfe',
        b'"sha256": "00000000',
        1,
    )
    assert replacement != trusted
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(trusted)
    original_read_bytes = Path.read_bytes
    reads = 0

    def replace_after_first_read(path):
        nonlocal reads
        payload = original_read_bytes(path)
        if path == manifest:
            reads += 1
            if reads == 1:
                manifest.write_bytes(replacement)
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_first_read)

    component = load_component_manifest(manifest)
    identity = component_receipt_identity(component)

    assert reads == 1
    assert identity["manifest_sha256"] == hashlib.sha256(trusted).hexdigest()


def test_bundled_asnumpy_source_snapshot_is_verified_and_receipted():
    manifest = json.loads(COMPONENT_MANIFEST.read_text(encoding='utf-8'))
    source = manifest['source_bundle']
    component = load_component_manifest(COMPONENT_MANIFEST)
    verified = verify_component_attachments(component)

    assert source['file_count'] == 198
    assert source['total_file_bytes'] == 1691974
    assert verified['source_bundle']['filename'] == source['filename']
    assert verified['source_bundle']['manifest_sha256'] == source['manifest_sha256']
    assert verified['source_bundle']['file_count'] == source['file_count']
    source_manifest = json.loads(
        (COMPONENT_MANIFEST.parent / source['manifest_filename']).read_text(encoding='utf-8')
    )
    assert source_manifest['commit'] == manifest['commit']
    assert source_manifest['tree'] == manifest['tree']
    assert len(source_manifest['files']) == source['file_count']
    assert component_receipt_identity(component)['source_bundle']['sha256'] == source['sha256']


def test_tampered_asnumpy_source_snapshot_fails_closed(tmp_path):
    component_dir = tmp_path / 'asnumpy'
    shutil.copytree(COMPONENT_MANIFEST.parent, component_dir)
    manifest = load_component_manifest(component_dir / 'manifest.json')
    source_name = manifest.source_bundle['filename']
    archive = component_dir / source_name
    archive.write_bytes(archive.read_bytes() + b'tamper')

    with pytest.raises(AsNumpyComponentError, match='source_bundle (?:byte count|SHA-256)'):
        verify_component_attachments(manifest)
