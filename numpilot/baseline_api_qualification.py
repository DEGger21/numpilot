"""Exact-profile qualifications for inherited AsNumpy baseline APIs.

This certificate is deliberately separate from gap-candidate qualifications.
The source bundle, integrated wheel, complete live Python dependency closure,
independent NumPy oracles and device traces are rechecked before admission.
Nothing in this module imports AsNumpy or upgrades its global runtime audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any, Mapping
import zipfile

from numpilot.capabilities import (CapabilityManifest, ExecutionProvenance,
    OperatorCallProfile, OperatorCapability, OperatorSignature, RuntimeIdentity,
    match_capability_profile, parse_profile_constraints)
from numpilot.gaps.official_qualification import (
    OFFICIAL_PRODUCER, apply_official_overlay, load_official_overlay,
)

SCHEMA = 'numpilot.baseline-api-qualification'
PRODUCER = 'numpilot-baseline-api-qualification-v1'
SAFE_BASELINE_APIS = frozenset({'absolute', 'add', 'tanh', 'square', 'mean',
    'not_equal', 'full', 'multiply', 'maximum', 'minimum', 'divide'})
BASELINE_MANIFEST_SHA256 = '408337443482fb9b614a83b8933d640192de15937ce7ccd440bf7b00218a7fa8'
PREFIXES = {name: 'aclnn' + suffix + '_' for name, suffix in {
    'absolute': 'Abs', 'add': 'Add', 'tanh': 'Tanh', 'square': 'Mul',
    'mean': 'Mean', 'not_equal': 'NeTensor', 'full': 'InplaceFillScalar',
    'multiply': 'Mul', 'maximum': 'Maximum', 'minimum': 'Minimum', 'divide': 'Div',
}.items()}
OP_TYPES = {'absolute': 'Abs', 'add': 'Add', 'divide': 'RealDiv', 'full': 'Fill',
    'maximum': 'Maximum', 'mean': 'ReduceMean', 'minimum': 'Minimum', 'multiply': 'Mul',
    'not_equal': 'NotEqual', 'square': 'Mul', 'tanh': 'Tanh'}
_REVIEWED_CONCAT_OVERLAY_SHA256 = {
    'bindings/python/bind_array.cpp': 'a9522ed3f6344d9feca4723ff32a16fdfb1eba91715d9d61f0efc5b69ea83c5e',
    'csrc/array/basic.cpp': '3459c333e2a2bb00b27fef570c400334b0b1b79b2cc818fe17c4b95d2988de53',
    'include/asnumpy/array/basic.hpp': '55c991b8d68cdc70feba327853f0a79ff1a1d6725015f0215c8fdb2de64b7484',
    'src/asnumpy/__init__.py': '295976c53a4fccc2807f9a3ef81821c063f839e6b32e4150d19288ee7af9224c',
    'src/asnumpy/array.py': '2e550d1de215182855a6746dad7c88210e7343d32b6ed284b08b9ffe6c81cfa4',
    'tests/asnumpy_tests/creation_tests/test_basic.py': 'f3bf01f12e4880b8e1b70a34e180a53c51171f25790769ad6bb47ee0b853f9e6',
}


DEPENDENCIES = (
    'src/asnumpy/math.py', 'src/asnumpy/_ufunc.py', 'src/asnumpy/array.py',
    'src/asnumpy/statistics.py', 'src/asnumpy/logic.py', 'src/asnumpy/utils.py',
    'bindings/python/bind_math.cpp', 'bindings/python/bind_array.cpp',
    'bindings/python/bind_statistics.cpp', 'bindings/python/bind_logic.cpp',
    'csrc/math/arithmetic_operations.cpp', 'csrc/math/floating_point_routines.cpp',
    'csrc/math/hyperbolic_functions.cpp', 'csrc/math/extrema_finding.cpp',
    'csrc/math/miscellaneous.cpp', 'csrc/array/basic.cpp',
    'csrc/statistics/averages_and_variances.cpp', 'csrc/logic/logic.cpp',
    'csrc/dtypes/promote.cpp', 'csrc/utils/cast.cpp', 'csrc/utils/npu_array.cpp',
    'include/asnumpy/utils/acl_executor.hpp', 'include/asnumpy/utils/dtype_promotion.hpp',
    'include/asnumpy/utils/npu_ops_macros.hpp',
)
_SOURCE_FIELDS = {'baseline_manifest', 'baseline_archive', 'candidate_tree',
                  'official_overlay', 'official_base_manifest', 'official_evidence_root'}
_EVIDENCE_FIELDS = {'worker', 'execution', 'numerical', 'unit_tests', 'traces'}
_SHA = re.compile(r'[0-9a-f]{64}\Z')


class BaselineApiQualificationError(ValueError):
    """A baseline certificate lacks the required exact-profile evidence."""


def _require(condition, message):
    if not condition:
        raise BaselineApiQualificationError(message)


def _canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _object(data, label):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate JSON key: ' + label)
            result[key] = value
        return result
    try:
        value = json.loads(data, object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise BaselineApiQualificationError('invalid JSON: ' + label) from exc
    _require(isinstance(value, dict), label + ' must be an object')
    return value


def _relative(value):
    _require(isinstance(value, str) and value and '\x00' not in value, 'invalid evidence path')
    path = PurePosixPath(value)
    _require(not path.is_absolute() and '..' not in path.parts and str(path) == value,
             'evidence path escapes root')
    return path


def _path(root, relative, *, directory=False):
    root = Path(root).absolute()
    path = root.joinpath(*_relative(relative).parts)
    for part in (path, *path.parents):
        _require(not part.is_symlink(), 'symlink evidence path rejected')
    _require(path.is_dir() if directory else path.is_file(), 'missing evidence: ' + str(path))
    return path


def _bound(root, ref, *, limit=128 * 1024 * 1024):
    _require(isinstance(ref, Mapping) and set(ref) == {'path', 'sha256'}, 'invalid file reference')
    _require(isinstance(ref['sha256'], str) and _SHA.fullmatch(ref['sha256']), 'invalid evidence SHA-256')
    path = _path(root, ref['path'])
    _require(path.stat().st_size <= limit, 'evidence file too large')
    value = path.read_bytes()
    _require(_sha(value) == ref['sha256'], 'evidence hash mismatch: ' + ref['path'])
    return value


def compute_baseline_qualification_id(data):
    clean = dict(data)
    clean.pop('qualification_id', None)
    return 'baseline-api-' + _sha(_canonical(clean))[:20]


@dataclass(frozen=True)
class BaselineApiEntry:
    registry_key: str
    qualified_api: str
    asnumpy_api: str
    constraints: tuple[str, ...]
    signature_variants: tuple[OperatorSignature, ...]
    profiles: tuple[OperatorCallProfile, ...]

    @classmethod
    def from_dict(cls, data):
        _require(isinstance(data, dict) and set(data) == {'registry_key', 'qualified_api',
            'asnumpy_api', 'constraints', 'signature_variants', 'profiles'}, 'unknown baseline entry fields')
        name = data['registry_key']
        _require(name in SAFE_BASELINE_APIS and data['qualified_api'] == 'numpy.' + name
                 and data['asnumpy_api'] == 'asnumpy.' + name, 'baseline API outside safe allowlist')
        constraints = tuple(data['constraints'])
        parsed = parse_profile_constraints(constraints)
        _require(parsed.get('dtype') == 'float32' and parsed.get('contiguous') is True,
                 'baseline entry requires contiguous float32')
        _require(parsed.get('rank', 0) >= 1 and 1 <= parsed.get('rank_max', 0) <= 3,
                 'baseline entry rank must be bounded within 1..3')
        signatures = tuple(OperatorSignature.from_dict(x) for x in data['signature_variants'])
        profiles = tuple(OperatorCallProfile.from_dict(x) for x in data['profiles'])
        _require(signatures and profiles and len(profiles) <= 128, 'baseline entry has no exact profiles')
        encoded = [_canonical(x.to_dict()) for x in profiles]
        _require(len(set(encoded)) == len(encoded), 'duplicate exact profile')
        capability = OperatorCapability(name=name, available=True, provenance=ExecutionProvenance.NATIVE_NPU,
            evidence='baseline certificate validation', constraints=constraints, signature_variants=signatures,
            qualified_profiles=profiles)
        for profile in profiles:
            _require(match_capability_profile(capability, profile).matched, 'profile violates baseline constraints')
            _require(all(all(extent > 0 for extent in t.shape) for t in profile.inputs), 'empty tensor profile rejected')
        return cls(name, data['qualified_api'], data['asnumpy_api'], constraints, signatures, profiles)

    def to_dict(self):
        return {'registry_key': self.registry_key, 'qualified_api': self.qualified_api,
            'asnumpy_api': self.asnumpy_api, 'constraints': list(self.constraints),
            'signature_variants': [x.to_dict() for x in self.signature_variants],
            'profiles': [x.to_dict() for x in self.profiles]}


@dataclass(frozen=True)
class BaselineApiCertificate:
    qualification_id: str
    runtime: RuntimeIdentity
    source: Mapping[str, Any]
    entries: tuple[BaselineApiEntry, ...]
    evidence: Mapping[str, Any]

    @property
    def entry_keys(self):
        return tuple(sorted(entry.registry_key for entry in self.entries))

    def match_profile(self, key, profile):
        return any(entry.registry_key == key and profile.to_dict() in [p.to_dict() for p in entry.profiles]
                   for entry in self.entries)

    def to_dict(self):
        return {'schema': SCHEMA, 'schema_version': 1, 'producer': PRODUCER,
            'qualification_id': self.qualification_id, 'runtime': self.runtime.to_dict(),
            'source': dict(self.source), 'entries': [x.to_dict() for x in self.entries],
            'evidence': dict(self.evidence)}

    @classmethod
    def from_dict(cls, data):
        _require(isinstance(data, dict) and set(data) == {'schema', 'schema_version', 'producer',
            'qualification_id', 'runtime', 'source', 'entries', 'evidence'}, 'unknown baseline certificate fields')
        _require(data['schema'] == SCHEMA and type(data['schema_version']) is int and data['schema_version'] == 1
                 and data['producer'] == PRODUCER, 'unknown baseline certificate schema/producer')
        _require(isinstance(data['source'], dict) and set(data['source']) == _SOURCE_FIELDS, 'invalid source proof fields')
        _require(isinstance(data['evidence'], dict) and set(data['evidence']) == _EVIDENCE_FIELDS, 'invalid evidence fields')
        runtime = RuntimeIdentity.from_dict(data['runtime'])
        _require(runtime.status in {'audit_required', 'matched'}, 'runtime status cannot be qualified')
        entries = tuple(BaselineApiEntry.from_dict(x) for x in data['entries'])
        names = [x.registry_key for x in entries]
        _require(names and names == sorted(set(names)), 'baseline entries must be unique and sorted')
        _require(data['qualification_id'] == compute_baseline_qualification_id(data), 'certificate content ID mismatch')
        return cls(data['qualification_id'], runtime, dict(data['source']), entries, dict(data['evidence']))


def load_baseline_api_certificate(path):
    path = Path(path).expanduser().absolute()
    raw = _bound(path.parent, {'path': path.name, 'sha256': _sha(path.read_bytes())}, limit=2 * 1024 * 1024)
    try:
        return BaselineApiCertificate.from_dict(_object(raw, 'baseline certificate'))
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise BaselineApiQualificationError(str(exc)) from exc


def baseline_api_certificate_sha256(certificate):
    """Stable serialized hash, matching producer indent=2/sort_keys/newline format."""
    return _sha((json.dumps(certificate.to_dict(), indent=2, sort_keys=True, ensure_ascii=False,
                           allow_nan=False) + '\n').encode('utf-8'))


# Reviewed stage-80 concat integration. These AST digests include signatures,
# decorators and the complete bodies, not only function names. A new integration
# with different source needs an explicit source-policy review before inheritance.
_REVIEWED_ARRAY_FUNCTION_AST_SHA256 = {
    'concatenate': '314b2ab34aefb21a44642e810bf07a331f7c3212ef1643dd0543abc938f8c07a',
    'vstack': 'c1fc8ab96d7198f074daa4e251fa32954d14f3bfb5b3b0c71500c195d545e23d',
    'hstack': '42446dac18a0c59e47203d7bcce0bb4f078539c54764ce6d128513b7e0da2f34',
}


def _verify_array_module_inheritance(original, candidate):
    """Protect constructor globals/imports as well as their function bodies."""
    before = ast.parse(original)
    after = ast.parse(candidate)
    def encode(node):
        return ast.dump(node, include_attributes=False)
    if encode(before) == encode(after):
        return
    originals = {node.name: node for node in before.body if isinstance(node, ast.FunctionDef)}
    allowed_imports = {'operator': 0, 'numpy': 0}
    old_import = ast.parse('from ._core.array import concatenate as _concatenate').body[0]
    new_import = ast.parse('from ._core.array import concatenate as _concatenate_legacy, concatenate_aclnn as _concatenate').body[0]
    converted_imports = 0
    normalized = []
    for node in after.body:
        if isinstance(node, ast.Import) and encode(node) in {
            encode(ast.parse('import operator').body[0]), encode(ast.parse('import numpy as np').body[0])}:
            module = node.names[0].name
            allowed_imports[module] += 1
            continue
        if encode(node) == encode(new_import):
            converted_imports += 1
            normalized.append(old_import)
            continue
        if isinstance(node, ast.FunctionDef) and node.name in _REVIEWED_ARRAY_FUNCTION_AST_SHA256:
            if encode(node) != encode(originals[node.name]):
                _require(_sha(encode(node).encode('utf-8')) == _REVIEWED_ARRAY_FUNCTION_AST_SHA256[node.name],
                         'unreviewed array function delta: ' + node.name)
                node = originals[node.name]
        normalized.append(node)
    _require(allowed_imports == {'operator': 1, 'numpy': 1} and converted_imports == 1,
             'array module import delta is not the reviewed concat integration')
    after.body = normalized
    _require(encode(after) == encode(before), 'inherited array module globals/imports changed')


def _verify_reviewed_source_overlay(modified):
    _require(set(modified) <= set(_REVIEWED_CONCAT_OVERLAY_SHA256),
             'source overlay changes paths outside the reviewed concat integration')
    _require(all(digest == _REVIEWED_CONCAT_OVERLAY_SHA256[name] for name, digest in modified.items()),
             'source overlay differs from the reviewed concat integration')


def _verify_source(certificate, root):
    proof = certificate.source
    manifest_bytes = _bound(root, proof['baseline_manifest'])
    _require(_sha(manifest_bytes) == BASELINE_MANIFEST_SHA256, 'baseline manifest is not author-pinned')
    manifest = _object(manifest_bytes, 'source manifest')
    expected = manifest['files']
    _require(isinstance(expected, dict) and len(expected) == 211, 'baseline must bind all 211 source files')
    archive_bytes = _bound(root, proof['baseline_archive'])
    _require(_sha(archive_bytes) == manifest['archive']['sha256'], 'baseline archive does not match manifest')
    prefix = manifest['archive']['prefix']
    originals = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r:gz') as archive:
        for member in archive:
            if member.isdir():
                continue
            _require(member.isfile() and member.name.startswith(prefix), 'invalid baseline archive member')
            name = member.name[len(prefix):]
            _relative(name)
            _require(name in expected and name not in originals, 'unexpected/duplicate source archive member')
            _require(member.size <= 8 * 1024 * 1024, 'source archive member too large')
            payload = archive.extractfile(member).read()
            _require(_sha(payload) == expected[name], 'baseline source member hash mismatch')
            originals[name] = payload
    _require(set(originals) == set(expected), 'baseline archive is incomplete')
    _bound(root, proof['official_overlay'])
    overlay = load_official_overlay(str(_path(root, proof['official_overlay']['path'])))
    base = CapabilityManifest.from_dict(_object(_bound(root, proof['official_base_manifest']), 'official base'))
    official_root = _path(root, proof['official_evidence_root'], directory=True)
    apply_official_overlay(base, overlay, evidence_root=str(official_root),
        trusted_producers=(OFFICIAL_PRODUCER,), current_identity=certificate.runtime)
    txn = _object(_bound(official_root, overlay.integration.transaction.to_dict()), 'integration transaction')
    binding = txn['source']['binding']
    _require(binding['source_manifest_sha256'] == BASELINE_MANIFEST_SHA256
             and binding['source_bundle_sha256'] == _sha(archive_bytes), 'transaction baseline binding mismatch')
    _require(txn['source']['file_count'] == 211 and txn['source']['revision'] == manifest['commit'],
             'transaction source count/revision mismatch')
    modified = dict(txn['source']['overlay_sha256'])
    _verify_reviewed_source_overlay(modified)
    _require(set(modified) <= set(expected), 'source overlay adds unbound source files')
    _require(txn['source']['overlay_before_sha256'] == {name: expected[name] for name in modified},
             'source overlay before-image mismatch')
    candidate = _path(root, proof['candidate_tree'], directory=True)
    contents = {}
    for name, baseline_hash in expected.items():
        path = _path(candidate, name)
        payload = path.read_bytes()
        _require(_sha(payload) == modified.get(name, baseline_hash), 'candidate source drift: ' + name)
        contents[name] = payload
    extras = [p for p in candidate.rglob('*') if p.is_file() and p.relative_to(candidate).as_posix() not in expected
              and p.relative_to(candidate).as_posix() != 'src/asnumpy/_version.py']
    _require(not extras, 'candidate source contains files outside the bound 211-file tree')
    # The shared array files changed to add concat. Their existing constructors
    # must not inherit qualification merely because the entire file was signed.
    changed = {'src/asnumpy/array.py', 'bindings/python/bind_array.cpp', 'csrc/array/basic.cpp'}
    for name in DEPENDENCIES:
        if name not in changed:
            _require(contents[name] == originals[name], 'inherited dependency changed: ' + name)
    _verify_array_module_inheritance(originals['src/asnumpy/array.py'], contents['src/asnumpy/array.py'])
    for function in ('full', 'full_like'):
        def node(payload):
            return ast.dump(next(n for n in ast.parse(payload).body if isinstance(n, ast.FunctionDef)
                                 and n.name == function), include_attributes=False)
        _require(node(contents['src/asnumpy/array.py']) == node(originals['src/asnumpy/array.py']),
                 'inherited creation Python function changed')
        pattern = rb'array\.def\("' + function.encode() + rb'"[^;]*;'
        _require(re.findall(pattern, contents['bindings/python/bind_array.cpp']) ==
                 re.findall(pattern, originals['bindings/python/bind_array.cpp']), 'inherited creation binding changed')
    for function in ('Full', 'Full_like'):
        pattern = rb'NPUArray ' + function.encode() + rb'\([^\n]*\).*?\n\}'
        before = re.findall(pattern, originals['csrc/array/basic.cpp'], re.S)
        after = re.findall(pattern, contents['csrc/array/basic.cpp'], re.S)
        _require(len(before) == len(after) == 1 and before == after, 'inherited creation C++ function changed')
    return overlay, official_root, txn


def _wheel_members(overlay, official_root, candidate_source):
    payload = _bound(official_root, overlay.integration.wheel.to_dict())
    members = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as wheel:
        names = wheel.namelist()
        _require(len(names) == len(set(names)), 'duplicate wheel member')
        for name in names:
            if name.startswith('asnumpy/') and (name.endswith('.py') or name.endswith('.so')):
                _relative(name)
                data = wheel.read(name)
                _require(len(data) <= 128 * 1024 * 1024, 'wheel member too large')
                members[name] = _sha(data)
                if name.endswith('.py') and name != 'asnumpy/_version.py':
                    source = _path(candidate_source, 'src/' + name)
                    _require(_sha(source.read_bytes()) == members[name],
                             'wheel Python module differs from proved candidate source: ' + name)
                # _version.py is generated by setuptools-scm during wheel build.
                # When retained in the candidate tree, that one permitted extra
                # must reproduce the wheel bytes, as well as the live closure.
                if name == 'asnumpy/_version.py' and (Path(candidate_source) / 'src' / name).exists():
                    generated = _path(candidate_source, 'src/' + name)
                    _require(_sha(generated.read_bytes()) == members[name],
                             'generated version metadata differs from wheel')
    _require({'asnumpy/__init__.py', 'asnumpy/_ufunc.py', 'asnumpy/_dtype.py'} <= set(members),
             'wheel dependency closure is missing')
    cores = [name for name in members if name.endswith('.so')]
    _require(len(cores) == 1 and members[cores[0]] == overlay.integration.runtime_core_sha256,
             'wheel native core mismatch')
    return members


def _verify_live_files(runtime, members):
    package = Path(runtime.module_path).parent
    actual = {}
    for path in package.rglob('*'):
        if path.suffix not in {'.py', '.so'}:
            continue
        relative = path.relative_to(package.parent).as_posix()
        value = _path(package.parent, relative).read_bytes()
        actual[relative] = _sha(value)
    _require(actual == members, 'live wheel Python/native dependency closure drift')


def _decode_argument(root, descriptor):
    import numpy as np
    _require(isinstance(descriptor, dict), 'invalid argument descriptor')
    kind = descriptor.get('kind')
    if kind == 'array':
        _require(set(descriptor) == {'kind', 'path', 'sha256'}, 'unknown array descriptor fields')
        data = _bound(root, {k: descriptor[k] for k in ('path', 'sha256')})
        value = np.load(io.BytesIO(data), allow_pickle=False)
        _require(isinstance(value, np.ndarray) and value.dtype == np.float32 and value.flags.c_contiguous
                 and 1 <= value.ndim <= 3 and all(x > 0 for x in value.shape), 'input outside fp32 tensor profile')
        return value
    _require(set(descriptor) == {'kind', 'value'}, 'unknown argument descriptor fields')
    value = descriptor['value']
    if kind == 'scalar':
        _require(type(value) in (int, float, bool) and (type(value) is not float or math.isfinite(value)),
                 'invalid finite scalar')
        return value
    if kind == 'shape':
        _require(isinstance(value, list) and 1 <= len(value) <= 3 and
                 all(type(x) is int and x > 0 for x in value), 'invalid creation shape')
        return tuple(value)
    if kind == 'dtype':
        _require(value == 'float32', 'unsupported dtype argument')
        return np.dtype('float32')
    raise BaselineApiQualificationError('unknown argument kind')


def _output(root, ref):
    import numpy as np
    value = np.load(io.BytesIO(_bound(root, ref)), allow_pickle=False)
    _require(isinstance(value, np.ndarray), 'numeric evidence must be an ndarray')
    return value


def _compare(expected, actual, label):
    import numpy as np
    _require(expected.shape == actual.shape and expected.dtype == actual.dtype, label + ' shape/dtype mismatch')
    _require(np.array_equal(np.isnan(expected), np.isnan(actual)), label + ' NaN mismatch')
    _require(np.array_equal(np.isposinf(expected), np.isposinf(actual)) and
             np.array_equal(np.isneginf(expected), np.isneginf(actual)), label + ' infinity mismatch')
    zeros = expected == 0
    _require(np.all(actual[zeros] == 0) and np.array_equal(np.signbit(expected[zeros]), np.signbit(actual[zeros])),
             label + ' signed-zero mismatch')
    if expected.dtype == np.bool_:
        _require(np.array_equal(expected, actual), label + ' boolean mismatch')
    else:
        _require(np.allclose(expected, actual, rtol=1e-5, atol=1e-6, equal_nan=True), label + ' numerical mismatch')


def _verify_profiler_row(name, row):
    duration = float(row['Task Duration(us)'])
    task_ok = row['Task Type'] in {'AI_VECTOR_CORE', 'AI_CORE'}
    # CANN 9.2 profiler stars_constant.SUBTASK_TYPE distinguishes MIX_AIV=7
    # from AI_CPU=12. ffts_pmu_calculator explicitly handles ACLNN MIX_AIV,
    # and aic/pmu_calculator processes it with AI-core/vector PMU records.
    # The observed ReduceMeanAiCore worker is the only extra admitted case.
    if row['Task Type'] == 'MIX_AIV':
        task_ok = name == 'mean' and row['Op Name'].startswith('aclnnMean_ReduceMeanAiCore_')
    _require(row['Op Name'].startswith(PREFIXES[name]) and row['OP Type'] == OP_TYPES[name]
             and row['Device_id'] == '0' and task_ok and math.isfinite(duration) and duration > 0,
             'invalid device profiler row')


def _verify_evidence(certificate, root, txn, members):
    import numpy as np
    from numpilot.capabilities import profile_api_call
    refs = certificate.evidence
    worker = _bound(root, refs['worker'])
    worker_sha = _sha(worker)
    numerical_bytes = _bound(root, refs['numerical'])
    numerical = _object(numerical_bytes, 'numerical evidence')
    execution = _object(_bound(root, refs['execution']), 'execution evidence')
    unit_bytes = _bound(root, refs['unit_tests'])
    unit = _object(unit_bytes, 'unit evidence')
    for name, doc in (('numerical', numerical), ('execution', execution), ('unit-tests', unit)):
        _require(doc.get('schema') == 'numpilot.baseline-api-' + name and
                 type(doc.get('schema_version')) is int and doc['schema_version'] == 1, 'unknown ' + name + ' schema')
        _require(doc.get('transaction_sha256') == txn['transaction_sha256'] and doc.get('worker_sha256') == worker_sha,
                 name + ' worker/transaction mismatch')
    _require(execution.get('status') == unit.get('status') == 'passed', 'execution/unit evidence failed')
    _require(execution.get('numerical_sha256') == _sha(numerical_bytes) and execution.get('unit_test_sha256') == _sha(unit_bytes),
             'execution artifact binding mismatch')
    _require(execution.get('runtime_before') == execution.get('runtime_after') == certificate.runtime.to_dict(),
             'execution runtime identity changed')
    _require(numerical.get('runtime_identity_sha256') == certificate.runtime.identity_sha256,
             'numerical runtime identity mismatch')
    _require(execution.get('runtime_files_sha256') == members, 'execution dependency closure mismatch')
    cases = numerical.get('cases')
    _require(isinstance(cases, list) and 1 <= len(cases) <= 512, 'missing/bounded numerical cases')
    entries = {e.registry_key: e for e in certificate.entries}
    coverage = {name: set() for name in entries}
    expected_execution = []
    case_ids = []
    for case in cases:
        _require(isinstance(case, dict) and set(case) == {'case_id', 'api', 'profile', 'arguments', 'keywords', 'expected', 'actual'},
                 'unknown numerical case fields')
        case_id, name = case['case_id'], case['api']
        _require(isinstance(case_id, str) and case_id and case_id not in case_ids, 'invalid/duplicate case ID')
        _require(name in entries, 'case API not in certificate')
        case_ids.append(case_id)
        args = tuple(_decode_argument(root, value) for value in case['arguments'])
        kwargs = {key: _decode_argument(root, value) for key, value in case['keywords'].items()}
        profile = profile_api_call(name, args, kwargs)
        _require(profile.to_dict() == case['profile'] and certificate.match_profile(name, profile),
                 'actual arguments do not match exact qualified profile')
        coverage[name].add(_canonical(case['profile']))
        with np.errstate(all='ignore'):
            oracle = np.asarray(getattr(np, name)(*args, **kwargs))
        _compare(oracle, _output(root, case['expected']), 'independent NumPy oracle')
        _compare(oracle, _output(root, case['actual']), 'device result')
        module = 'statistics' if name == 'mean' else 'array' if name == 'full' else 'logic' if name == 'not_equal' else 'math'
        expected_execution.append({'case_id': case_id, 'api': name, 'profile': case['profile'],
            'native_calls': 1, 'host_fallback_calls': 0, 'target': '_core.' + module + '.' + name})
    for name, entry in entries.items():
        _require(coverage[name] == {_canonical(x.to_dict()) for x in entry.profiles}, 'untested exact profile: ' + name)
    _require(execution.get('cases') == expected_execution and all(type(row.get('native_calls')) is int
             and type(row.get('host_fallback_calls')) is int for row in execution['cases']),
             'execution dispatch evidence mismatch')
    _require(unit.get('cases') == [{'case_id': x, 'status': 'passed'} for x in case_ids], 'unit case coverage mismatch')
    workers = execution.get('workers')
    _require(isinstance(workers, list) and len(workers) == len(entries), 'missing per-API worker receipts')
    worker_refs = {}
    for worker_ref in workers:
        _require(isinstance(worker_ref, dict) and set(worker_ref) == {'api', 'path', 'sha256'}, 'invalid worker reference')
        api = worker_ref['api']
        _require(api in entries and api not in worker_refs, 'duplicate worker API')
        ref = {key: worker_ref[key] for key in ('path', 'sha256')}
        worker_refs[api] = ref
        child = _object(_bound(root, ref), 'per-API execution')
        _require(child.get('schema') == 'numpilot.baseline-api-execution' and type(child.get('schema_version')) is int
                 and child['schema_version'] == 1 and child.get('status') == 'passed' and child.get('api') == api,
                 'invalid per-API worker schema/API/status')
        for key in ('transaction_sha256', 'worker_sha256', 'runtime_before', 'runtime_after', 'runtime_files_sha256'):
            _require(child.get(key) == execution.get(key), 'per-API worker binding drift: ' + key)
        _require(child.get('cases') == [row for row in expected_execution if row['api'] == api]
                 and all(type(row.get('native_calls')) is int and type(row.get('host_fallback_calls')) is int
                         for row in child['cases']), 'per-API dispatch mismatch')
        child_numerical_bytes = _bound(root, child.get('numerical'))
        child_unit_bytes = _bound(root, child.get('unit_tests'))
        _require(child.get('numerical_sha256') == _sha(child_numerical_bytes)
                 and child.get('unit_test_sha256') == _sha(child_unit_bytes), 'per-API evidence hashes mismatch')
        child_num = _object(child_numerical_bytes, 'per-API numerical')
        _require(type(child_num.get('schema_version')) is int, 'invalid per-API numerical schema version')
        for key in ('schema', 'schema_version', 'transaction_sha256', 'worker_sha256', 'runtime_identity_sha256'):
            _require(child_num.get(key) == numerical.get(key), 'per-API numerical binding drift')
        subset = [row for row in cases if row['api'] == api]
        _require(child_num.get('cases') == subset, 'per-API numerical coverage mismatch')
        child_unit = _object(child_unit_bytes, 'per-API unit')
        _require(type(child_unit.get('schema_version')) is int, 'invalid per-API unit schema version')
        for key in ('schema', 'schema_version', 'transaction_sha256', 'worker_sha256', 'status'):
            _require(child_unit.get(key) == unit.get(key), 'per-API unit binding drift')
        _require(child_unit.get('cases') == [{'case_id': row['case_id'], 'status': 'passed'} for row in subset],
                 'per-API unit case mismatch')
    traces = refs['traces']
    _require(isinstance(traces, list) and len(traces) == len(entries), 'missing per-API profiler traces')
    seen = set()
    trace_paths = set()
    trace_hashes = set()
    for trace in traces:
        _require(isinstance(trace, dict) and set(trace) == {'api', 'files', 'execution'}, 'invalid trace group')
        name = trace['api']
        _require(name in entries and name not in seen and isinstance(trace['files'], list) and trace['files'],
                 'duplicate/missing trace API')
        seen.add(name)
        _require(trace['execution'] == worker_refs[name], 'trace/worker binding mismatch')
        count = 0
        paths = set()
        for ref in trace['files']:
            _require(ref.get('path') not in paths, 'duplicate profiler file')
            paths.add(ref.get('path'))
            _require(ref.get('path') not in trace_paths, 'profiler file reused across API groups')
            trace_paths.add(ref.get('path'))
            trace_bytes = _bound(root, ref)
            trace_sha = _sha(trace_bytes)
            _require(trace_sha not in trace_hashes, 'profiler bytes reused across trace groups')
            trace_hashes.add(trace_sha)
            rows = csv.DictReader(io.StringIO(trace_bytes.decode('utf-8-sig')))
            _require({'Op Name', 'OP Type', 'Task Type', 'Device_id', 'Task Duration(us)'} <= set(rows.fieldnames or []),
                     'unknown profiler CSV columns')
            for row in rows:
                if row['Op Name'].startswith(PREFIXES[name]):
                    _verify_profiler_row(name, row)
                    count += 1
        _require(count >= sum(c['api'] == name for c in cases), 'insufficient actual native kernel evidence: ' + name)


def validate_baseline_api_certificate(certificate, *, evidence_root, current_identity,
                                      verify_live_files=True):
    """Recheck all evidence; live callers additionally verify every wheel module."""
    try:
        certificate = BaselineApiCertificate.from_dict(certificate.to_dict())
        current = current_identity.to_dict() if hasattr(current_identity, 'to_dict') else current_identity
        current = RuntimeIdentity.from_dict(current)
        _require(current.to_dict() == certificate.runtime.to_dict(), 'baseline live runtime identity drift')
        overlay, official_root, txn = _verify_source(certificate, Path(evidence_root))
        members = _wheel_members(overlay, official_root, _path(evidence_root, certificate.source['candidate_tree'], directory=True))
        _verify_evidence(certificate, Path(evidence_root), txn, members)
        if verify_live_files:
            _verify_live_files(current, members)
        return certificate
    except BaselineApiQualificationError:
        raise
    except (ValueError, TypeError, KeyError, OSError, AttributeError, RuntimeError) as exc:
        raise BaselineApiQualificationError('baseline evidence rejected: ' + str(exc)) from exc


def apply_baseline_api_certificate(base, certificate, *, evidence_root, current_identity,
                                   verify_live_files=True):
    certificate = validate_baseline_api_certificate(certificate, evidence_root=evidence_root,
        current_identity=current_identity, verify_live_files=verify_live_files)
    operators = dict(base.operators)
    for entry in certificate.entries:
        existing = operators.get(entry.registry_key)
        _require(existing is not None and existing.available, 'qualified baseline API absent from runtime')
        expected = 'baseline API qualification ' + certificate.qualification_id
        _require(existing.provenance in {ExecutionProvenance.UNKNOWN, ExecutionProvenance.NATIVE_NPU},
                 'baseline certificate cannot replace a different provenance')
        _require(existing.provenance is ExecutionProvenance.UNKNOWN or existing.evidence == expected,
                 'baseline certificate conflicts with existing qualification')
        operators[entry.registry_key] = OperatorCapability(name=entry.registry_key, available=True,
            provenance=ExecutionProvenance.NATIVE_NPU, evidence=expected, constraints=entry.constraints,
            signature_variants=entry.signature_variants, qualified_profiles=entry.profiles)
    suffix = '+baseline-api-' + certificate.qualification_id.removeprefix('baseline-api-')
    return CapabilityManifest(backend=base.backend, backend_version=base.backend_version,
        operators=operators, environment=dict(base.environment), schema_version=base.schema_version,
        audit_id=base.audit_id if base.audit_id.endswith(suffix) else base.audit_id + suffix)
