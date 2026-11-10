"""CPU-only baseline certificates: real profiles, NumPy oracles and bound files."""
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import numpy as np
import pytest

import numpilot.baseline_api_qualification as bq
from numpilot.capabilities import (CapabilityManifest, ExecutionProvenance,
    OperatorCapability, RuntimeIdentity, match_capability_profile, profile_api_call)
from tests.test_scoped_qualification import case as scope_case


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    path.chmod(0o600)


def ref(root, path):
    return {'path': Path(path).relative_to(root).as_posix(), 'sha256': sha(Path(path).read_bytes())}


def identity():
    return RuntimeIdentity(asnumpy_repository='unknown', asnumpy_commit='unknown', asnumpy_tree='unknown',
        asnumpy_version='0.4.1', wrapper_sha256={'__init__.py':'a'*64,'array.py':'b'*64},
        module_path='/runtime/asnumpy/__init__.py', module_sha256='a'*64,
        native_module_path='/runtime/asnumpy/_core.so', native_module_sha256='c'*64,
        cann_home='/cann', cann_version='9.2.0', soc_model='Ascend910B3',
        python_executable='/bin/python3', python_version='3.12.0',
        bootstrap_receipt_path='/receipt.json', bootstrap_receipt_sha256='unknown',
        audit_id='unaudited-asnumpy-version', status='audit_required', issues=('not globally audited',))


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path/'qualification'
    root.mkdir(mode=0o700)
    (root/'source').mkdir()
    runtime = identity()
    tx = '9'*64
    (root/'worker.py').write_text('# synthetic CPU test evidence; never executed\n')
    worker_sha = sha((root/'worker.py').read_bytes())
    members = {'asnumpy/__init__.py':'a'*64,'asnumpy/_core.so':'c'*64,
               'asnumpy/_ufunc.py':'d'*64,'asnumpy/_dtype.py':'e'*64}
    a = np.array([[-0., 0., 1.], [2., -3., 4.]], dtype=np.float32)
    c = np.array([[-0., 0., 4.], [5., -6., 7.]], dtype=np.float32)
    profile = profile_api_call('add', (a, c), {}).to_dict()
    args = []
    for name, value in [('a',a),('b',c)]:
        path = root/(name+'.npy')
        np.save(path,value,allow_pickle=False)
        args.append({'kind':'array', **ref(root,path)})
    for name in ('expected','actual'):
        np.save(root/(name+'.npy'),np.add(a,c),allow_pickle=False)
    probe={'case_id':'add-probe','api':'add','profile':profile,'arguments':args,'keywords':{},
           'expected':ref(root,root/'expected.npy'),'actual':ref(root,root/'actual.npy')}
    numerical={'schema':'numpilot.baseline-api-numerical','schema_version':1,
        'transaction_sha256':tx,'worker_sha256':worker_sha,
        'runtime_identity_sha256':runtime.identity_sha256,'cases':[probe]}
    unit={'schema':'numpilot.baseline-api-unit-tests','schema_version':1,'status':'passed',
        'transaction_sha256':tx,'worker_sha256':worker_sha,'cases':[{'case_id':'add-probe','status':'passed'}]}
    native={'case_id':'add-probe','api':'add','profile':profile,'native_calls':1,
            'host_fallback_calls':0,'target':'_core.math.add'}
    dump(root/'api/add/numerical.json',numerical)
    dump(root/'api/add/unit-tests.json',unit)
    child={'schema':'numpilot.baseline-api-execution','schema_version':1,'status':'passed','api':'add',
        'transaction_sha256':tx,'worker_sha256':worker_sha,'runtime_before':runtime.to_dict(),
        'runtime_after':runtime.to_dict(),'runtime_files_sha256':members,'cases':[native],
        'numerical':ref(root,root/'api/add/numerical.json'),'unit_tests':ref(root,root/'api/add/unit-tests.json'),
        'numerical_sha256':sha((root/'api/add/numerical.json').read_bytes()),
        'unit_test_sha256':sha((root/'api/add/unit-tests.json').read_bytes())}
    dump(root/'api/add/execution.json',child)
    execution={k:v for k,v in child.items() if k not in ('api','numerical','unit_tests')}
    execution['workers']=[{'api':'add',**ref(root,root/'api/add/execution.json')}]
    dump(root/'numerical.json',numerical)
    dump(root/'unit-tests.json',unit)
    dump(root/'execution.json',execution)
    (root/'trace.csv').write_text('Op Name,OP Type,Task Type,Device_id,Task Duration(us)\naclnnAdd_Add_0,Add,AI_VECTOR_CORE,0,1.25\n')
    data={'schema':bq.SCHEMA,'schema_version':1,'producer':bq.PRODUCER,'qualification_id':'pending',
        'runtime':runtime.to_dict(), 'source':{name:{} for name in bq._SOURCE_FIELDS},
        'entries':[{'registry_key':'add','qualified_api':'numpy.add','asnumpy_api':'asnumpy.add',
            'constraints':['dtype=float32','rank>=1','rank<=3','contiguous=true'],
            'signature_variants':[{'positional_args':2,'keyword_args':[]}],'profiles':[profile]}],
        'evidence':{'worker':ref(root,root/'worker.py'),'execution':ref(root,root/'execution.json'),
            'numerical':ref(root,root/'numerical.json'),'unit_tests':ref(root,root/'unit-tests.json'),
            'traces':[{'api':'add','execution':ref(root,root/'api/add/execution.json'),
                       'files':[ref(root,root/'trace.csv')]}]}}
    data['source']['candidate_tree']='source'
    data['qualification_id']=bq.compute_baseline_qualification_id(data)
    monkeypatch.setattr(bq,'_verify_source',lambda cert,root:(None,root,{'transaction_sha256':tx}))
    monkeypatch.setattr(bq,'_wheel_members',lambda overlay,root,candidate:members)
    return SimpleNamespace(root=root,data=data,runtime=runtime,numerical=numerical,unit=unit,
                           execution=execution,child=child,members=members,profile=profile)


def cert(bundle):
    bundle.data['qualification_id']=bq.compute_baseline_qualification_id(bundle.data)
    return bq.BaselineApiCertificate.from_dict(bundle.data)


def validate(bundle):
    return bq.validate_baseline_api_certificate(cert(bundle),evidence_root=bundle.root,
                                               current_identity=bundle.runtime,verify_live_files=False)


def reseal(bundle):
    """Update content hashes after an intentional internally-consistent tamper."""
    root=bundle.root
    dump(root/'api/add/numerical.json',bundle.numerical)
    dump(root/'api/add/unit-tests.json',bundle.unit)
    bundle.child['numerical']=ref(root,root/'api/add/numerical.json')
    bundle.child['unit_tests']=ref(root,root/'api/add/unit-tests.json')
    bundle.child['numerical_sha256']=bundle.child['numerical']['sha256']
    bundle.child['unit_test_sha256']=bundle.child['unit_tests']['sha256']
    dump(root/'api/add/execution.json',bundle.child)
    bundle.execution['numerical_sha256']=bundle.child['numerical_sha256']
    bundle.execution['unit_test_sha256']=bundle.child['unit_test_sha256']
    bundle.execution['workers']=[{'api':'add',**ref(root,root/'api/add/execution.json')}]
    dump(root/'numerical.json',bundle.numerical)
    dump(root/'unit-tests.json',bundle.unit)
    dump(root/'execution.json',bundle.execution)
    for key,name in [('execution','execution.json'),('numerical','numerical.json'),('unit_tests','unit-tests.json')]:
        bundle.data['evidence'][key]=ref(root,root/name)
    bundle.data['evidence']['traces'][0]['execution']=ref(root,root/'api/add/execution.json')


def test_complete_numpy_and_profile_evidence_is_accepted_without_device(bundle):
    value=validate(bundle)
    assert value.entry_keys==('add',)
    assert value.runtime.status=='audit_required'
    base=CapabilityManifest(backend='npu',backend_version='0.4.1',environment={},audit_id='unaudited',
        operators={'add':OperatorCapability(name='add',available=True,provenance=ExecutionProvenance.UNKNOWN,evidence='unqualified')})
    admitted=bq.apply_baseline_api_certificate(base,value,evidence_root=bundle.root,
        current_identity=bundle.runtime,verify_live_files=False)
    cap=admitted.operators['add']
    assert match_capability_profile(cap,profile_api_call('add',(np.ones((2,3),dtype='f'),)*2,{})).matched
    assert not match_capability_profile(cap,profile_api_call('add',(np.ones((3,3),dtype='f'),)*2,{})).matched


@pytest.mark.parametrize('field,value',[('schema','unknown'),('schema_version',True),('producer','external')])
def test_unknown_schema_producer_rejected(bundle,field,value):
    bundle.data[field]=value
    with pytest.raises(bq.BaselineApiQualificationError):
        validate(bundle)


@pytest.mark.parametrize('api',['copysign','power','subtract','full_like'])
def test_unsafe_or_unselected_api_cannot_be_certified(bundle,api):
    entry=bundle.data['entries'][0]
    entry.update(registry_key=api,qualified_api='numpy.'+api,asnumpy_api='asnumpy.'+api)
    with pytest.raises(bq.BaselineApiQualificationError,match='allowlist'):
        validate(bundle)


def test_unprobed_profile_is_rejected(bundle):
    profile=profile_api_call('add',(np.ones((3,3),dtype='f'),)*2,{}).to_dict()
    bundle.data['entries'][0]['profiles'].append(profile)
    with pytest.raises(bq.BaselineApiQualificationError,match='untested'):
        validate(bundle)


def test_rehashed_fake_expected_is_recomputed_independently(bundle):
    wrong=np.ones((2,3),dtype=np.float32)
    np.save(bundle.root/'expected.npy',wrong,allow_pickle=False)
    bundle.numerical['cases'][0]['expected']=ref(bundle.root,bundle.root/'expected.npy')
    reseal(bundle)
    with pytest.raises(bq.BaselineApiQualificationError,match='oracle'):
        validate(bundle)


def test_zero_sign_is_not_hidden_by_allclose(bundle):
    actual=np.load(bundle.root/'actual.npy')
    assert np.signbit(actual[0,0])
    actual[0,0]=0.0
    np.save(bundle.root/'actual.npy',actual,allow_pickle=False)
    bundle.numerical['cases'][0]['actual']=ref(bundle.root,bundle.root/'actual.npy')
    reseal(bundle)
    with pytest.raises(bq.BaselineApiQualificationError,match='signed-zero'):
        validate(bundle)


def test_core_dispatch_not_just_a_passed_status(bundle):
    bundle.child['cases'][0]['native_calls']=0
    reseal(bundle)
    with pytest.raises(bq.BaselineApiQualificationError,match='dispatch'):
        validate(bundle)


def test_child_worker_transaction_is_bound(bundle):
    bundle.child['transaction_sha256']='8'*64
    reseal(bundle)
    with pytest.raises(bq.BaselineApiQualificationError,match='worker binding'):
        validate(bundle)


@pytest.mark.parametrize('value',['0','nan','-1','inf'])
def test_bad_profiler_duration_rejected(bundle,value):
    path=bundle.root/'trace.csv'
    path.write_text(path.read_text().replace('1.25',value))
    bundle.data['evidence']['traces'][0]['files']=[ref(bundle.root,path)]
    with pytest.raises(bq.BaselineApiQualificationError,match='profiler row'):
        validate(bundle)


def test_a_different_aclnn_api_does_not_prove_add(bundle):
    path=bundle.root/'trace.csv'
    path.write_text(path.read_text().replace('aclnnAdd_','aclnnMul_'))
    bundle.data['evidence']['traces'][0]['files']=[ref(bundle.root,path)]
    with pytest.raises(bq.BaselineApiQualificationError,match='kernel evidence'):
        validate(bundle)


def test_runtime_dependency_closure_includes_ufunc_and_extra_code(tmp_path):
    package=tmp_path/'asnumpy'
    package.mkdir()
    members={}
    for name in ('__init__.py','_ufunc.py','_dtype.py','_core.so'):
        (package/name).write_bytes(name.encode())
        members['asnumpy/'+name]=sha(name.encode())
    runtime=SimpleNamespace(module_path=str(package/'__init__.py'))
    bq._verify_live_files(runtime,members)
    (package/'_ufunc.py').write_text('drift')
    with pytest.raises(bq.BaselineApiQualificationError,match='closure drift'):
        bq._verify_live_files(runtime,members)
    (package/'_ufunc.py').write_text('_ufunc.py')
    (package/'unexpected.py').write_text('import hook')
    with pytest.raises(bq.BaselineApiQualificationError,match='closure drift'):
        bq._verify_live_files(runtime,members)


def test_source_proof_rechecks_all_211_files_and_inherited_creation(tmp_path,monkeypatch):
    # Author-pinned artifacts are distributed in the repository, not server-private evidence.
    component=Path(__file__).resolve().parents[1]/'components/asnumpy'
    manifest_path=component/'source-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    archive_path=component/manifest['archive']['filename']
    root=tmp_path/'bundle'
    root.mkdir(mode=0o700)
    (root/'baseline.json').write_bytes(manifest_path.read_bytes())
    (root/'baseline.tar.gz').write_bytes(archive_path.read_bytes())
    candidate=root/'source'
    candidate.mkdir()
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if member.isfile():
                relative=member.name[len(manifest['archive']['prefix']):]
                path=candidate/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    official=root/'official'
    official.mkdir()
    txn={'source':{'binding':{'source_manifest_sha256':bq.BASELINE_MANIFEST_SHA256,
        'source_bundle_sha256':manifest['archive']['sha256']},'file_count':211,
        'revision':manifest['commit'],'overlay_sha256':{},'overlay_before_sha256':{}}}
    dump(official/'transaction.json',txn)
    dump(root/'overlay.json',{})
    base=CapabilityManifest(backend='npu',backend_version='0.4.1',operators={},environment={},audit_id='base')
    dump(root/'base.json',base.to_dict())
    class FileRef:
        def to_dict(self): return ref(official,official/'transaction.json')
    overlay=SimpleNamespace(integration=SimpleNamespace(transaction=FileRef()))
    monkeypatch.setattr(bq,'load_official_overlay',lambda path:overlay)
    monkeypatch.setattr(bq,'apply_official_overlay',lambda *args,**kwargs:base)
    proof={'baseline_manifest':ref(root,root/'baseline.json'),'baseline_archive':ref(root,root/'baseline.tar.gz'),
        'candidate_tree':'source','official_overlay':ref(root,root/'overlay.json'),
        'official_base_manifest':ref(root,root/'base.json'),'official_evidence_root':'official'}
    certificate=SimpleNamespace(source=proof,runtime=identity())
    bq._verify_source(certificate,root)
    (candidate/'src/asnumpy/_ufunc.py').write_text('drift')
    with pytest.raises(bq.BaselineApiQualificationError,match='source drift'):
        bq._verify_source(certificate,root)


def test_scope_v2_combines_baseline_with_official_without_promoting_runtime(scope_case,bundle,monkeypatch):
    import numpilot.scoped_qualification as sq
    # The same runtime has both the official concat and exact-profile add receipts.
    bundle.runtime=RuntimeIdentity.from_dict(scope_case.identity)
    bundle.data['runtime']=scope_case.identity
    for execution in (bundle.execution,bundle.child):
        execution['runtime_before']=execution['runtime_after']=scope_case.identity
    bundle.numerical['runtime_identity_sha256']=bundle.runtime.identity_sha256
    reseal(bundle)
    certificate=cert(bundle)
    certificate_path=bundle.root/'baseline-api-certificate.json'
    dump(certificate_path,certificate.to_dict())
    cert_sha=sha(certificate_path.read_bytes())
    base=scope_case.manifest
    raw_add=OperatorCapability(name='add',available=True,provenance=ExecutionProvenance.UNKNOWN,evidence='unknown')
    combined=CapabilityManifest(backend=base.backend,backend_version=base.backend_version,
        operators={**base.operators,'add':raw_add},environment=base.environment,audit_id=base.audit_id)
    monkeypatch.setattr(sq,'_load_official_admission',lambda root,official_root,before:(scope_case.overlay,combined))
    admission_path=scope_case.root/'overlay-admission.json'
    admission=json.loads(admission_path.read_text())
    admission.update(operators=['add','concatenate'],qualified_names=['numpy.add','numpy.concatenate'],baseline_api={
        'qualification_id':certificate.qualification_id,'certificate_sha256':cert_sha,
        'runtime_identity_sha256':bundle.runtime.identity_sha256,'operators':['add'],'qualified_names':['numpy.add']})
    dump(admission_path,admission)
    scope_case.validation['backend_provenance']['audit_id'] += '+'+certificate.qualification_id
    dump(scope_case.root/'validation-evidence.json',scope_case.validation)
    scope_ref=sq.build_scoped_qualification(scope_case.root,**scope_case.kwargs,
        baseline_certificate_path=certificate_path,baseline_evidence_root=bundle.root)
    assert scope_ref['schema_version']==2
    assert scope_ref['baseline_certificate_sha256']==cert_sha
    validated=sq.validate_scoped_qualification(scope_ref,expected_runtime_identity=scope_case.identity)
    assert validated['runtime_identity_status']=='audit_required'
    assert validated['required_ops']==['concatenate']
    assert validated['global_runtime_audit_promoted'] is False
    # A different baseline cannot be smuggled into an already sealed sample.
    scope_ref['baseline_qualification_id']='baseline-api-'+'f'*20
    with pytest.raises(sq.ScopedQualificationError,match='baseline_qualification_id'):
        sq.validate_scoped_qualification(scope_ref)


def test_performance_reuse_must_keep_baseline_certificate_identity(monkeypatch):
    from numpilot.performance_evidence import _compatible_scopes,PerformanceEvidenceError
    import numpilot.scoped_performance as perf
    base={key:'same' for key in ('sample_id','source_sha256','analysis_source_sha256',
        'generated_source_sha256','profile_sha256','runtime_identity_sha256','overlay_sha256','qualification_id','required_ops')}
    produced={**base,'baseline_qualification_id':'baseline-api-a','baseline_certificate_sha256':'a'*64}
    current={**base,'baseline_qualification_id':'baseline-api-b','baseline_certificate_sha256':'b'*64}
    monkeypatch.setattr(perf,'load_scope',lambda ref:produced if ref=={'which':'produced'} else current)
    with pytest.raises(PerformanceEvidenceError,match='baseline scope mismatch'):
        _compatible_scopes({'which':'produced'},{'which':'current'})


def test_array_globals_rebinding_cannot_inherit_full_source():
    import ast
    original = (Path(__file__).resolve().parents[1]/'components/asnumpy')
    manifest=json.loads((original/'source-manifest.json').read_text())
    with tarfile.open(original/manifest['archive']['filename']) as archive:
        source=archive.extractfile(manifest['archive']['prefix']+'src/asnumpy/array.py').read()
    bq._verify_array_module_inheritance(source,source)
    # full's AST remains identical; its native dependency is silently replaced.
    for suffix in (b'\n_full = lambda *args: None\n', b'\nndarray = object\n',
                   b'\n_normalize_shape = lambda shape: [1]\n'):
        with pytest.raises(bq.BaselineApiQualificationError):
            bq._verify_array_module_inheritance(source,source+suffix)


def test_wheel_python_must_equal_proved_source_except_generated_version(tmp_path):
    import zipfile
    source=tmp_path/'source'
    package=source/'src/asnumpy'
    package.mkdir(parents=True)
    payloads={'asnumpy/__init__.py':b'# init\n','asnumpy/_ufunc.py':b'# ufunc\n',
              'asnumpy/_dtype.py':b'# dtype\n','asnumpy/_version.py':b'version="built"\n',
              'asnumpy/_core.so':b'ELF core'}
    for name,data in payloads.items():
        (source/'src'/name).write_bytes(data)
    (package/'_version.py').unlink()
    wheel=tmp_path/'candidate.whl'
    def write_wheel():
        with zipfile.ZipFile(wheel,'w') as archive:
            for name,data in payloads.items(): archive.writestr(name,data)
    class FileRef:
        def to_dict(self): return ref(tmp_path,wheel)
    overlay=SimpleNamespace(integration=SimpleNamespace(wheel=FileRef(),runtime_core_sha256=sha(payloads['asnumpy/_core.so'])))
    write_wheel()
    bq._wheel_members(overlay,tmp_path,source)
    payloads['asnumpy/_ufunc.py']=b'# different runtime dispatch\n'
    write_wheel()
    with pytest.raises(bq.BaselineApiQualificationError,match='proved candidate source'):
        bq._wheel_members(overlay,tmp_path,source)


def test_only_reviewed_concat_overlay_can_inherit_source():
    bq._verify_reviewed_source_overlay(dict(bq._REVIEWED_CONCAT_OVERLAY_SHA256))
    for changed in ({'CMakeLists.txt':'a'*64}, {'include/asnumpy/utils/acl_resource.hpp':'a'*64},
                    {'csrc/array/basic.cpp':'a'*64}):
        with pytest.raises(bq.BaselineApiQualificationError,match='reviewed concat'):
            bq._verify_reviewed_source_overlay(changed)


@pytest.mark.parametrize('api,op_type,task_type,valid',[
    ('mean','ReduceMean','MIX_AIV',True),
    ('mean','ReduceMean','AI_CPU',False),
    ('mean','Add','MIX_AIV',False),
    ('add','Add','MIX_AIV',False),
    ('add','Mul','AI_VECTOR_CORE',False),
])
def test_profiler_mix_aiv_is_only_the_observed_mean_compute_kernel(api,op_type,task_type,valid):
    row={'Op Name':'aclnnMean_ReduceMeanAiCore_ReduceMean' if api=='mean' else 'aclnnAdd_Add_0',
         'OP Type':op_type,'Task Type':task_type,'Device_id':'0','Task Duration(us)':'1.25'}
    if valid:
        bq._verify_profiler_row(api,row)
    else:
        with pytest.raises(bq.BaselineApiQualificationError,match='profiler row'):
            bq._verify_profiler_row(api,row)


def test_renamed_identical_csv_cannot_duplicate_device_evidence(bundle):
    alias=bundle.root/'renamed.csv'
    alias.write_bytes((bundle.root/'trace.csv').read_bytes())
    bundle.data['evidence']['traces'][0]['files'].append(ref(bundle.root,alias))
    with pytest.raises(bq.BaselineApiQualificationError,match='profiler bytes reused'):
        validate(bundle)
