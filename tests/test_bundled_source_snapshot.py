"""The isolated build must preserve exactly the author-shipped source bytes."""

import hashlib
import json
import tarfile

import pytest

from numpilot.asnumpy_component import (
    default_component_root,
    load_component_manifest,
    verify_component_attachments,
)
from numpilot.gaps import integration
from numpilot.gaps.bundled_source import snapshot_bundled_source


@pytest.fixture
def bundled_reference(tmp_path):
    component = load_component_manifest(default_component_root())
    verify_component_attachments(component)
    bundle = component.source_bundle
    manifest = json.loads((component.root / bundle['manifest_filename']).read_text())
    expected = manifest['files']
    root = tmp_path / 'reference'
    root.mkdir()
    with tarfile.open(component.root / bundle['filename'], 'r:gz') as archive:
        # Construct only manifest members instead of extracting arbitrary paths.
        for relative in expected:
            member = archive.getmember(manifest['archive']['prefix'] + relative)
            assert member.isfile()
            output = root / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.extractfile(member).read())
    return root, component, manifest


def test_snapshot_preserves_bundled_matmul_and_ignores_unlisted_files(bundled_reference, tmp_path):
    reference, component, manifest = bundled_reference
    # Untracked files, build products and unrelated local state cannot enter a build.
    (reference / 'untracked.py').write_text('raise RuntimeError("not part of the component")\n')
    (reference / 'build').mkdir()
    (reference / 'build' / 'untrusted.so').write_bytes(b'not-a-library')
    (reference / '.git').mkdir()
    (reference / '.git' / 'config').write_text('[core]\n')
    target = tmp_path / 'staging'
    binding = {}
    revision, hashes, total = snapshot_bundled_source(reference, target, binding=binding)
    assert binding == {
        'kind': 'verified_component_bundle',
        'component_id': component.component_id,
        'component_manifest_sha256': component.manifest_sha256,
        'source_bundle_sha256': component.source_bundle['sha256'],
        'source_manifest_sha256': component.source_bundle['manifest_sha256'],
        'repository': component.repository,
        'commit': component.commit,
        'tree': component.tree,
    }
    assert revision == component.commit
    assert hashes == manifest['files']
    assert len(hashes) == manifest['archive']['file_count']
    assert total == manifest['archive']['total_file_bytes']
    assert not (target / 'untracked.py').exists()
    assert not (target / 'build').exists()
    assert not (target / '.git').exists()
    for relative in ('src/asnumpy/linalg/direct.py', 'csrc/linalg/product.cpp'):
        assert (target / relative).read_bytes() == (reference / relative).read_bytes()
        assert hashlib.sha256((target / relative).read_bytes()).hexdigest() == hashes[relative]
    assert {str(path.relative_to(target)) for path in target.rglob('*') if path.is_file()} == set(hashes)


def test_snapshot_materializes_internal_leaf_alias(bundled_reference, tmp_path):
    reference, _, manifest = bundled_reference
    original = (reference / 'AGENTS.md').read_bytes()
    assert original == (reference / 'CLAUDE.md').read_bytes()
    (reference / 'AGENTS.md').unlink()
    (reference / 'AGENTS.md').symlink_to('CLAUDE.md')
    target = tmp_path / 'staging'
    _, hashes, _ = snapshot_bundled_source(reference, target)
    assert hashes['AGENTS.md'] == manifest['files']['AGENTS.md']
    assert not (target / 'AGENTS.md').is_symlink()
    assert (target / 'AGENTS.md').read_bytes() == original


@pytest.mark.parametrize('mutation', ['byte_drift', 'missing', 'outside_link', 'directory_link'])
def test_snapshot_rejects_drift_without_partial_target(bundled_reference, tmp_path, mutation):
    reference, _, _ = bundled_reference
    relative = 'src/asnumpy/linalg/direct.py'
    source = reference / relative
    if mutation == 'byte_drift':
        source.write_bytes(source.read_bytes() + b'\n# unapproved matmul patch\n')
    elif mutation == 'missing':
        source.unlink()
    elif mutation == 'outside_link':
        external = tmp_path / 'external.py'
        external.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(external)
    else:
        original = reference / 'src'
        alias_target = reference / 'src-materialized'
        original.rename(alias_target)
        original.symlink_to(alias_target, target_is_directory=True)
    target = tmp_path / 'staging'
    with pytest.raises((ValueError, OSError)):
        snapshot_bundled_source(reference, target)
    assert not target.exists()


@pytest.mark.parametrize('condition', ['reference_link', 'existing_target'])
def test_snapshot_rejects_linked_root_or_existing_target(bundled_reference, tmp_path, condition):
    reference, _, _ = bundled_reference
    target = tmp_path / 'staging'
    if condition == 'reference_link':
        alias = tmp_path / 'reference-alias'
        alias.symlink_to(reference, target_is_directory=True)
        reference = alias
    else:
        target.mkdir()
        (target / 'sentinel').write_text('keep')
    with pytest.raises(ValueError):
        snapshot_bundled_source(reference, target)
    if condition == 'existing_target':
        assert (target / 'sentinel').read_text() == 'keep'


@pytest.mark.parametrize('code', ['source_dirty', 'source_unavailable'])
def test_integration_falls_back_only_to_verified_bundle(bundled_reference, tmp_path, monkeypatch, code):
    reference, component, manifest = bundled_reference

    def dirty_identity(*_args):
        raise integration.IntegrationError(code, 'temporary reference state')

    monkeypatch.setattr(integration, '_git_identity', dirty_identity)
    target = tmp_path / 'staging'
    binding = {}
    revision, hashes, _ = integration._source_snapshot(reference, target, 30, binding=binding)
    assert binding['kind'] == 'verified_component_bundle'
    assert binding['source_manifest_sha256'] == component.source_bundle['manifest_sha256']
    assert revision == component.commit
    assert hashes == manifest['files']
    assert (target / 'src/asnumpy/linalg/direct.py').read_bytes() == (reference / 'src/asnumpy/linalg/direct.py').read_bytes()


def test_integration_dirty_reference_drift_reports_binding_mismatch(bundled_reference, tmp_path, monkeypatch):
    reference, _, _ = bundled_reference
    (reference / 'src/asnumpy/linalg/direct.py').write_text('unapproved source\n')

    def dirty_identity(*_args):
        raise integration.IntegrationError('source_dirty', 'dirty')

    monkeypatch.setattr(integration, '_git_identity', dirty_identity)
    target = tmp_path / 'staging'
    with pytest.raises(integration.IntegrationError) as exc:
        integration._source_snapshot(reference, target, 30)
    assert exc.value.code == 'source_binding_mismatch'
    assert not target.exists()


def test_integration_does_not_mask_other_git_errors(tmp_path, monkeypatch):
    def failed_identity(*_args):
        raise integration.IntegrationError('subprocess_failed', 'git could not run')

    monkeypatch.setattr(integration, '_git_identity', failed_identity)
    with pytest.raises(integration.IntegrationError) as exc:
        integration._source_snapshot(tmp_path, tmp_path / 'staging', 30)
    assert exc.value.code == 'subprocess_failed'


def test_clean_git_snapshot_identifies_its_source_kind(tmp_path, monkeypatch):
    reference = tmp_path / 'reference'
    reference.mkdir()
    (reference / 'source.cpp').write_text('// clean source\n')
    monkeypatch.setattr(integration, '_git_identity', lambda *_: 'a' * 40)
    binding = {}
    revision, hashes, total = integration._source_snapshot(
        reference, tmp_path / 'staging', 30, binding=binding,
    )
    assert binding == {'kind': 'clean_git', 'revision': 'a' * 40}
    assert revision == binding['revision']
    assert hashes == {'source.cpp': hashlib.sha256(b'// clean source\n').hexdigest()}
    assert total == len(b'// clean source\n')
