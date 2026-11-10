"""CPU-only producer checks; fake native modules are never NPU evidence."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from numpilot.capabilities import profile_api_call
from tools import qualify_baseline_apis as tool


@pytest.fixture
def prepared(tmp_path):
    cases, sources = tool.prepare_cases(tmp_path)
    return tmp_path, cases, sources


def test_prepare_uses_real_three_sources_and_every_safe_api(prepared):
    root, cases, sources = prepared
    assert [source['sample_id'] for source in sources] == list(tool.SAMPLES)
    for source in sources:
        path = tool.REPO / 'benchmark/cases' / source['sample_id'] / 'program.py'
        assert source['sha256'] == tool.sha(path)
    assert {case['api'] for case in cases} == set(tool.APIS)
    source_cases = [case for case in cases if ':baseline-case:' in case['case_id']]
    # This count binds the actual static-lowering workload, including both adds
    # in G001 and all five constant arrays in D1005.
    assert Counter(case['api'] for case in source_cases) == {
        'absolute': 2, 'add': 4, 'divide': 1, 'full': 5, 'maximum': 1,
        'mean': 2, 'minimum': 1, 'multiply': 2, 'not_equal': 1,
        'square': 2, 'tanh': 3,
    }
    assert all(source_case['expected']['path'].startswith('data/') for source_case in source_cases)


def test_all_saved_arguments_replay_exact_profile_and_numpy_oracle(prepared):
    root, cases, _ = prepared
    for case in cases:
        args = tuple(tool.decode(root, item) for item in case['arguments'])
        kwargs = {name: tool.decode(root, item) for name, item in case['keywords'].items()}
        assert profile_api_call(case['api'], args, kwargs).to_dict() == case['profile']
        expected_ref = case['expected']
        expected_path = root / expected_ref['path']
        assert tool.sha(expected_path) == expected_ref['sha256']
        expected = np.load(expected_path, allow_pickle=False)
        with np.errstate(all='ignore'):
            oracle = np.asarray(getattr(np, case['api'])(*args, **kwargs))
        tool.compare(expected, oracle)


def test_creation_has_output_metadata_without_fake_input(prepared):
    root, cases, _ = prepared
    full = [case for case in cases if case['api'] == 'full']
    assert len(full) == 5
    observed = []
    for case in full:
        args = tuple(tool.decode(root, item) for item in case['arguments'])
        profile = case['profile']
        assert profile['inputs'] == []
        assert profile['creation_output']['dtype'] == 'float32'
        assert profile['creation_output']['shape'] == list(args[0])
        assert profile['creation_fill']['kind'] == 'python:float'
        assert case['keywords'] == {'dtype': {'kind': 'dtype', 'value': 'float32'}}
        observed.append((args[0], args[1]))
    assert observed == [((4, 8), .5), ((4, 8), .25), ((4, 8), 0.),
                        ((4, 8), 1.), ((4, 16), 1.)]


def test_noncreation_exact_profiles_have_independent_edges(prepared):
    root, cases, _ = prepared
    originals = {}
    for case in cases:
        if ':baseline-case:' in case['case_id'] and case['api'] != 'full':
            key = (case['api'], json.dumps(case['profile'], sort_keys=True))
            originals[key] = case
    for (api, encoded), original in originals.items():
        for label in ('nonfinite', 'zero_signs'):
            edge = [case for case in cases if case['api'] == api
                    and ':' + label + ':' in case['case_id']
                    and json.dumps(case['profile'], sort_keys=True) == encoded]
            assert len(edge) == 1
            tensors = [tool.decode(root, item) for item in edge[0]['arguments'] if item['kind'] == 'array']
            assert tensors and all(value.dtype == np.float32 and value.flags.c_contiguous for value in tensors)
            if label == 'nonfinite':
                assert np.isnan(tensors[0]).any()
                assert np.isposinf(tensors[0]).any()
                assert np.isneginf(tensors[0]).any()
            else:
                assert (tensors[0] == 0).all()
                assert np.signbit(tensors[0]).any()
                assert (~np.signbit(tensors[0])).any()
                if api == 'divide':
                    assert (tensors[1] == 1).all()


def test_real_nan_case_oracle_preserves_nan_and_infinity_classification(prepared):
    root, cases, _ = prepared
    case = next(case for case in cases if ':baseline-case:I_gap_model/I003_nan_classification:' in case['case_id'])
    args = [tool.decode(root, item) for item in case['arguments']]
    assert np.signbit(args[0][1])
    assert np.isnan(args[0][2])
    assert np.isposinf(args[0][3]) and np.isneginf(args[0][4])
    np.testing.assert_array_equal(np.load(root / case['expected']['path']),
                                  np.array([False, False, True, False, False, False, False]))


def test_prepare_restores_numpy_on_source_failure(tmp_path, monkeypatch):
    originals = {name: getattr(np, name) for name in tool.APIS}
    monkeypatch.setattr(tool, 'REPO', tmp_path / 'missing-repo')
    with pytest.raises(FileNotFoundError):
        tool.prepare_cases(tmp_path)
    for name, original in originals.items():
        assert getattr(np, name) is original


@pytest.mark.parametrize('expected,actual,reason', [
    ([0.], [-0.], 'signed zero'), ([-0.], [0.], 'signed zero'),
    ([0.], [1e-9], 'zero value'), ([np.inf], [-np.inf], 'nonfinite'),
    ([np.nan], [0.], 'nonfinite'), ([-np.inf], [np.nan], 'nonfinite'),
    ([1.], [2.], None),
])
def test_compare_rejects_semantic_errors(expected, actual, reason):
    with pytest.raises(AssertionError, match=reason):
        tool.compare(np.array(expected, np.float32), np.array(actual, np.float32))


@pytest.mark.parametrize('values,dtype', [
    ([0., -0., np.nan, np.inf, -np.inf, .25], np.float32),
    ([False, True, False], np.bool_),
])
def test_compare_accepts_equal_special_values(values, dtype):
    array = np.array(values, dtype=dtype)
    tool.compare(array, array.copy())


@pytest.mark.parametrize('actual', [np.zeros((1, 2), np.float32), np.zeros(2, np.float64)])
def test_compare_requires_exact_shape_and_dtype(actual):
    with pytest.raises(AssertionError, match='shape/dtype'):
        tool.compare(np.zeros(2, np.float32), actual)


def test_array_decode_checks_saved_bytes(tmp_path):
    encoded = tool.encode(tmp_path, 'input', np.array([1., -0.], np.float32))
    decoded = tool.decode(tmp_path, encoded)
    assert decoded.dtype == np.float32 and np.signbit(decoded[1])
    with (tmp_path / encoded['path']).open('ab') as handle:
        handle.write(b'tamper')
    with pytest.raises(ValueError, match='input hash'):
        tool.decode(tmp_path, encoded)


@pytest.mark.parametrize('value', [np.dtype('float64'), np.array([object()], dtype=object),
                                  (), (0, 1), (True, 1), complex(1), np.nan, np.inf])
def test_encode_rejects_unsupported_input_metadata(tmp_path, value):
    with pytest.raises((TypeError, ValueError)):
        tool.encode(tmp_path, 'argument', value)


def _fake_native(monkeypatch, name, *, bad_alias=False, bad_loop=False, empty=False):
    calls = []
    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return 'native-result'
    module_name = 'statistics' if name == 'mean' else 'array' if name == 'full' else 'logic' if name == 'not_equal' else 'math'
    public = SimpleNamespace(**{'_' + name: object() if bad_alias else native})
    if name == 'add':
        loops = [] if empty else [SimpleNamespace(routine=native), SimpleNamespace(routine=object() if bad_loop else native)]
        public.add = SimpleNamespace(_ops=SimpleNamespace(ops=loops), _fallback=lambda *args: 'host')
    core = SimpleNamespace(**{module_name: SimpleNamespace(**{name: native})})
    modules = {'asnumpy.' + module_name: public, 'asnumpy._core': core}
    monkeypatch.setattr(tool.importlib, 'import_module', lambda key: modules[key])
    return public, calls


@pytest.mark.parametrize('name', ['absolute', 'full', 'mean', 'not_equal'])
def test_native_probe_counts_real_core_alias_without_importing_asnumpy(monkeypatch, name):
    public, calls = _fake_native(monkeypatch, name)
    counts, target = tool.install_probe(None, name)
    assert counts == {'native': 0, 'fallback': 0}
    assert getattr(public, '_' + name)(1, axis=2) == 'native-result'
    assert calls == [((1,), {'axis': 2})]
    assert counts == {'native': 1, 'fallback': 0}
    assert target.endswith('.' + name)


def test_add_probe_replaces_captured_dtype_loop_and_forbids_fallback(monkeypatch):
    public, calls = _fake_native(monkeypatch, 'add')
    counts, target = tool.install_probe(None, 'add')
    for loop in public.add._ops.ops:
        assert loop.routine(1, 2) == 'native-result'
    assert counts['native'] == 2 and len(calls) == 2
    assert target == '_core.math.add'
    with pytest.raises(ValueError, match='fallback'):
        public.add._fallback(1, 2)
    assert counts['fallback'] == 1


@pytest.mark.parametrize('kwargs,reason', [({'bad_alias': True}, 'native alias'),
                                        ({'bad_loop': True}, 'dtype loop'),
                                        ({'empty': True}, 'dtype loop')])
def test_probe_rejects_unknown_add_routes(monkeypatch, kwargs, reason):
    _fake_native(monkeypatch, 'add', **kwargs)
    with pytest.raises(ValueError, match=reason):
        tool.install_probe(None, 'add')


def test_runtime_files_requires_live_wheel_bytes_and_rejects_symlinks(tmp_path):
    site = tmp_path / 'site'
    (site / 'asnumpy').mkdir(parents=True)
    files = {'asnumpy/__init__.py': b'# fixture\n', 'asnumpy/_core.so': b'fixture-native', 'other.py': b'ignored'}
    wheel = tmp_path / 'fixture.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
            if name.startswith('asnumpy/'):
                (site / name).write_bytes(data)
    assert set(tool.runtime_files(site, wheel)) == {'asnumpy/__init__.py', 'asnumpy/_core.so'}
    core = site / 'asnumpy/_core.so'
    core.write_bytes(b'changed')
    with pytest.raises(ValueError, match='wheel dependency'):
        tool.runtime_files(site, wheel)
    core.unlink()
    other = tmp_path / 'elsewhere.so'
    other.write_bytes(files['asnumpy/_core.so'])
    core.symlink_to(other)
    with pytest.raises(ValueError, match='wheel dependency'):
        tool.runtime_files(site, wheel)


def _publish_fixture(root, monkeypatch, statuses):
    """Only the independent full-certificate verifier is stubbed here.

    These records test producer aggregation policy; they do not constitute
    validated native, source, numerical or profiler evidence.
    """
    from numpilot import baseline_api_qualification as contract
    monkeypatch.setattr(tool, 'APIS', tuple(statuses))
    runtime = {'identity_sha256': 'b' * 64, 'status': 'audit_required'}
    tool.write(root / 'plan.json', {'transaction_sha256': 'c' * 64})
    for name in ('worker.py', 'baseline-source-manifest.json', 'baseline-source.tar.gz',
                 'official/capability-overlay.json', 'official-base-manifest.json'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('CPU aggregate fixture')
    profiles = {}
    for name, status in statuses.items():
        folder = root / 'apis' / name
        profile = profile_api_call(name, (np.ones(4, np.float32),) * (2 if name == 'add' else 1), {}).to_dict()
        profiles[name] = profile
        tool.write(folder / 'execution.json', {'status': status, 'runtime_before': deepcopy(runtime),
            'runtime_files_sha256': {'asnumpy/_core.so': 'a' * 64}, 'cases': [{'api': name, 'profile': profile}]})
        if status != 'passed':
            continue
        tool.write(folder / 'numerical.json', {'cases': [{'api': name, 'profile': profile}]})
        tool.write(folder / 'unit-tests.json', {'cases': [{'case_id': name, 'status': 'passed'}]})
        trace = root / 'traces' / name / 'op_summary_fixture.csv'
        trace.parent.mkdir(parents=True)
        trace.write_text('CPU fixture,not a device trace\n')
    validated = []
    monkeypatch.setattr(contract, 'BaselineApiCertificate', SimpleNamespace(from_dict=lambda value: value))
    def verifier(certificate, **kwargs):
        validated.append(deepcopy(certificate))
    monkeypatch.setattr(contract, 'validate_baseline_api_certificate', verifier)
    return validated


def test_publish_excludes_entire_failed_api_before_certificate_verification(tmp_path, monkeypatch):
    validated = _publish_fixture(tmp_path, monkeypatch, {'add': 'passed', 'tanh': 'failed'})
    tool.publish(tmp_path)
    assert len(validated) == 1
    certificate = json.loads((tmp_path / 'baseline-api-certificate.json').read_text())
    assert [entry['registry_key'] for entry in certificate['entries']] == ['add']
    assert [trace['api'] for trace in certificate['evidence']['traces']] == ['add']
    for name in ('numerical.json', 'execution.json'):
        assert {case['api'] for case in json.loads((tmp_path / name).read_text())['cases']} == {'add'}
    receipt = json.loads((tmp_path / 'production-receipt.json').read_text())
    assert receipt['qualified_apis'] == ['add'] and receipt['excluded_apis'] == ['tanh']
    assert receipt['global_runtime_status'] == 'audit_required' and receipt['model_called'] is False


def test_publish_refuses_all_failed_apis(tmp_path, monkeypatch):
    validated = _publish_fixture(tmp_path, monkeypatch, {'add': 'failed', 'tanh': 'failed'})
    with pytest.raises(ValueError, match='no APIs passed'):
        tool.publish(tmp_path)
    assert validated == []
    assert not (tmp_path / 'baseline-api-certificate.json').exists()


def test_publish_refuses_cross_worker_identity_drift(tmp_path, monkeypatch):
    validated = _publish_fixture(tmp_path, monkeypatch, {'add': 'passed', 'tanh': 'passed'})
    path = tmp_path / 'apis/tanh/execution.json'
    value = json.loads(path.read_text())
    value['runtime_before']['identity_sha256'] = 'd' * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='cross-process runtime drift'):
        tool.publish(tmp_path)
    assert validated == [] and not (tmp_path / 'baseline-api-certificate.json').exists()


def test_publish_never_writes_certificate_if_independent_verifier_rejects(tmp_path, monkeypatch):
    from numpilot import baseline_api_qualification as contract
    _publish_fixture(tmp_path, monkeypatch, {'add': 'passed'})
    def rejected(*args, **kwargs):
        raise ValueError('invalid source or native evidence')
    monkeypatch.setattr(contract, 'validate_baseline_api_certificate', rejected)
    with pytest.raises(ValueError, match='invalid source'):
        tool.publish(tmp_path)
    assert not (tmp_path / 'baseline-api-certificate.json').exists()
    assert not (tmp_path / 'production-receipt.json').exists()


def test_worker_injects_explicit_backend_extractor_for_device_arrays(tmp_path, monkeypatch):
    """Execute the real worker logic with isolated CPU doubles, never AsNumpy.

    The fake device intentionally has familiar dtype/shape attributes. The
    generic extractor must reject it; only the explicit backend extractor can
    identify this runtime's device type and reproduce the captured host profile.
    """
    import sys
    from types import ModuleType
    from numpilot.backends import npu as backend_module
    from numpilot.capabilities import tensor_profile_from_value

    class DeviceArray:
        def __init__(self, host):
            self.host = np.asarray(host).copy()
            self.dtype, self.shape = self.host.dtype, self.host.shape
        @classmethod
        def from_numpy(cls, host):
            return cls(host)
        def to_numpy(self):
            return self.host.copy()

    extractor_calls = []
    runtime = {'identity_sha256': 'a' * 64, 'status': 'audit_required'}
    class CpuBackendDouble:
        def runtime_identity(self):
            return SimpleNamespace(to_dict=lambda: deepcopy(runtime))
        def _extract_tensor_profile(self, value):
            extractor_calls.append(value)
            return tensor_profile_from_value(value.host) if type(value) is DeviceArray else None

    host = np.array([-1., -0., 3.], dtype=np.float32)
    assert profile_api_call('absolute', (DeviceArray(host),), {}).unprofiled_operands
    profile = profile_api_call('absolute', (host,), {}).to_dict()
    case = {'case_id': '0000:cpu-extractor-fixture:absolute', 'api': 'absolute', 'profile': profile,
            'arguments': [tool.encode(tmp_path, 'input', host)], 'keywords': {},
            'expected': tool.save_array(tmp_path, 'expected.npy', np.absolute(host))}
    site = tmp_path / 'isolated-site'
    tool.write(tmp_path / 'plan.json', {'runtime_site': str(site), 'wheel': str(tmp_path / 'fixture.whl'),
        'transaction_sha256': 'b' * 64, 'cases': [case]})
    counts = {'native': 0, 'fallback': 0}
    fake_asnumpy = ModuleType('asnumpy')
    fake_asnumpy.__file__ = str(site / 'asnumpy/__init__.py')
    fake_asnumpy.ndarray = DeviceArray
    def absolute(value):
        counts['native'] += 1
        return DeviceArray(np.absolute(value.host))
    fake_asnumpy.absolute = absolute
    monkeypatch.setitem(sys.modules, 'asnumpy', fake_asnumpy)
    monkeypatch.setattr(backend_module, 'NpuBackend', CpuBackendDouble)
    monkeypatch.setattr(tool, 'install_probe', lambda ap, name: (counts, '_core.math.absolute'))
    monkeypatch.setattr(tool, 'runtime_files', lambda site, wheel: {'fixture-only': 'c' * 64})
    def get_count(pointer):
        pointer._obj.value = 1
        return 0
    def synchronize():
        return 0
    monkeypatch.setattr(tool.ctypes, 'CDLL', lambda name: SimpleNamespace(
        aclrtGetDeviceCount=get_count, aclrtSynchronizeDevice=synchronize))
    # Avoid removing the real test runner's logging sinks.
    monkeypatch.setitem(sys.modules, 'loguru', SimpleNamespace(logger=SimpleNamespace(remove=lambda: None)))
    tool.worker(SimpleNamespace(out=str(tmp_path), worker_api='absolute'))
    execution = json.loads((tmp_path / 'apis/absolute/execution.json').read_text())
    assert execution['api'] == 'absolute' and execution['status'] == 'passed'
    assert execution['cases'][0]['profile'] == profile
    assert execution['cases'][0]['native_calls'] == 1
    assert execution['cases'][0]['host_fallback_calls'] == 0
    assert len(extractor_calls) == 1 and type(extractor_calls[0]) is DeviceArray


def test_write_identical_opt_in_preserves_original_bytes_and_stat(tmp_path):
    path = tmp_path / 'aggregate.json'
    value = {'b': [1, 2], 'a': {'passed': True}}
    tool.write(path, value)
    before_bytes, before_stat = path.read_bytes(), path.stat()
    tool.write(path, {'a': {'passed': True}, 'b': [1, 2]}, allow_identical=True)
    assert path.read_bytes() == before_bytes
    assert path.stat().st_ino == before_stat.st_ino
    assert path.stat().st_mtime_ns == before_stat.st_mtime_ns
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('allow_identical,new_value', [(False, {'value': 1}), (True, {'value': 2})])
def test_write_does_not_overwrite_existing_json(tmp_path, allow_identical, new_value):
    path = tmp_path / 'evidence.json'
    tool.write(path, {'value': 1})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        tool.write(path, new_value, allow_identical=allow_identical)
    assert path.read_bytes() == original


def test_write_semantically_equal_but_different_bytes_is_not_reused(tmp_path):
    path = tmp_path / 'evidence.json'
    original = b'{"value":1}\n'
    path.write_bytes(original)
    with pytest.raises(FileExistsError):
        tool.write(path, {'value': 1}, allow_identical=True)
    assert path.read_bytes() == original


@pytest.mark.parametrize('dangling', [False, True])
@pytest.mark.parametrize('allow_identical', [False, True])
def test_write_rejects_symbolic_link_even_when_target_is_identical(tmp_path, dangling, allow_identical):
    target, link = tmp_path / 'target.json', tmp_path / 'link.json'
    value = {'value': 1}
    if not dangling:
        tool.write(target, value)
    link.symlink_to(target)
    before = target.read_bytes() if target.exists() else None
    with pytest.raises(FileExistsError):
        tool.write(link, value, allow_identical=allow_identical)
    assert link.is_symlink()
    assert (target.read_bytes() if target.exists() else None) == before


def test_publish_reuses_only_identical_aggregates_after_verifier_failure(tmp_path, monkeypatch):
    from numpilot import baseline_api_qualification as contract
    validated = _publish_fixture(tmp_path, monkeypatch, {'add': 'passed'})
    real_fixture_verifier = contract.validate_baseline_api_certificate
    raw_paths = [tmp_path / 'apis/add' / name for name in ('execution.json', 'numerical.json', 'unit-tests.json')]
    raw_paths += [tmp_path / 'worker.py', tmp_path / 'traces/add/op_summary_fixture.csv']
    before_raw = {path: path.read_bytes() for path in raw_paths}
    def rejected(*args, **kwargs):
        raise ValueError('fixture rejection before publication')
    monkeypatch.setattr(contract, 'validate_baseline_api_certificate', rejected)
    with pytest.raises(ValueError, match='before publication'):
        tool.publish(tmp_path)
    aggregate_paths = [tmp_path / name for name in ('execution.json', 'numerical.json', 'unit-tests.json')]
    before_aggregate = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in aggregate_paths}
    assert not (tmp_path / 'baseline-api-certificate.json').exists()
    monkeypatch.setattr(contract, 'validate_baseline_api_certificate', real_fixture_verifier)
    tool.publish(tmp_path)
    assert len(validated) == 1
    for path, content in before_raw.items():
        assert path.read_bytes() == content
    for path, (content, mtime) in before_aggregate.items():
        assert path.read_bytes() == content and path.stat().st_mtime_ns == mtime
    certificate = (tmp_path / 'baseline-api-certificate.json').read_bytes()
    receipt = (tmp_path / 'production-receipt.json').read_bytes()
    with pytest.raises(FileExistsError):
        tool.publish(tmp_path)
    assert (tmp_path / 'baseline-api-certificate.json').read_bytes() == certificate
    assert (tmp_path / 'production-receipt.json').read_bytes() == receipt


def test_publish_refuses_changed_existing_aggregate(tmp_path, monkeypatch):
    _publish_fixture(tmp_path, monkeypatch, {'add': 'passed'})
    path = tmp_path / 'numerical.json'
    tool.write(path, {'changed': True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        tool.publish(tmp_path)
    assert path.read_bytes() == original
    assert not (tmp_path / 'baseline-api-certificate.json').exists()
