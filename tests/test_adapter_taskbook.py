"""CPU-only checks that local and remote adapters receive the persisted request."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import pytest

from numpilot.agents.analysis import AnalysisAgent
from numpilot.contracts import AnalysisResult, OpUse
from numpilot.gap_requirements import build_gap_taskbook
from numpilot.gaps.adapter_handoff import load_adapter_handoff, write_adapter_handoff
from numpilot.gaps.provider import DraftGenerationError, LlmDraftProvider
from numpilot.gaps.spec import build_gap_specs
from numpilot.engines.llm import LlmEngine
from tests.test_gap_completion import (
    _FakeSkillBundle, _adapter_context_plan, _configured_status, _remote_receipt,
)


def _book(api='cbrt', dtype='float32'):
    args='x, x' if api == 'matmul' else 'x'
    source='import numpy as np\ndef forward(x):\n    return np.%s(%s)\n' % (api,args)
    # A static fixture gap, independent of the host's capability installation.
    call='np.%s(%s)' % (api,args)
    use=OpUse(api,3,False,reason='API 不在静态覆盖集',col_offset=11,
              qualified_name='numpy.'+api,call_expr=call,positional_args=2 if api == 'matmul' else 1)
    analysis=AnalysisResult('case.py',[use],[api],False)
    spec=build_gap_specs(source,analysis,target_soc='Ascend910B3',cann_version='9.2.0-beta.2')[0]
    return build_gap_taskbook(replace(spec,dtype_rules=(dtype,)),source=source)


def _adapter_inputs(tmp_path,book):
    context={'CMakeLists.txt':'project(asnumpy)\n'}
    paths=('csrc/custom_ops/registry/cbrt/op.yaml',
           'tests/asnumpy_tests/custom_ops_tests/test_cbrt.py')
    spec=book['gap_spec']
    return dict(operator_artifact={'artifact_id':'draft-fixture','gap_id':spec['gap_id'],
                                  'qualified_api':spec['qualified_api'],'source_sha256':spec['source_sha256']},
                adapter_contract=json.dumps({'api_name':spec['api_name'],'gap_id':spec['gap_id'],
                    'qualified_api':spec['qualified_api'],'source_sha256':spec['source_sha256']}),
                asnumpy_context=context,expected_paths=paths,
                adapter_context_plan=_adapter_context_plan(tmp_path,context,paths),gap_taskbook=book)


def test_adapter_taskbook_persisted_before_first_callback_and_present_every_stage(tmp_path,monkeypatch):
    out=tmp_path/'adapter';out.mkdir()
    book=_book()
    seen=[]
    def remote_chat(_system,prompt,**_kwargs):
        persisted=json.loads((out/'taskbook/GAP_TASKBOOK.json').read_text())
        assert persisted==book
        md=(out/'taskbook/GAP_TASKBOOK.md').read_text()
        assert md in prompt
        assert book['taskbook_id'] in prompt
        path=re.search(r'Return exactly `([^`]+)`',prompt).group(1)
        seen.append(path)
        response='\n'.join(('NUMPILOT_BUNDLE_V1','SUMMARY: fixture','FILE: '+path,'CONTENT:',
                             '{}' if path.endswith('.yaml') else 'def test_fixture():\n    pass',
                             'END_FILE','END_BUNDLE'))
        return response,_remote_receipt(response)
    monkeypatch.setattr(LlmEngine,'_chat_with_receipt',staticmethod(remote_chat))
    provider=LlmDraftProvider(max_attempts=1,configuration_status=_configured_status(tmp_path),
                              skill_bundle=_FakeSkillBundle(),staged_direct=True)
    report=provider.generate_asnumpy_adapter_candidate(out,**_adapter_inputs(tmp_path,book))
    assert len(seen)==2
    assert report['status']=='generated_unverified'


def test_non_fp32_adapter_taskbook_is_saved_before_provider_resolution_and_blocked(tmp_path,monkeypatch):
    out=tmp_path/'adapter';out.mkdir()
    book=_book(dtype='float64')
    provider=LlmDraftProvider(max_attempts=1,skill_bundle=_FakeSkillBundle())
    monkeypatch.setattr(provider,'_resolve_chat',lambda:pytest.fail('dtype conflict reached model configuration'))
    with pytest.raises(DraftGenerationError,match='float32 taskbook'):
        provider.generate_asnumpy_adapter_candidate(out,**_adapter_inputs(tmp_path,book))
    assert json.loads((out/'taskbook/GAP_TASKBOOK.json').read_text())==book
    assert (out/'taskbook/GAP_TASKBOOK.md').is_file()
    assert not (out/'candidate').exists()


def _handoff(tmp_path, *, book=None, artifact_updates=None, contract_updates=None):
    book=book or _book('matmul')
    path='src/asnumpy/linalg/direct.py'
    content='def matmul(a, b):\n    return a @ b\n'
    plan={'expected_overlay_paths':[path],
          'context_sha256':{path:hashlib.sha256(content.encode()).hexdigest()}}
    spec=book['gap_spec']
    root=tmp_path/'handoff'
    artifact={'artifact_id':'candidate-fixture','gap_id':spec['gap_id'],
              'qualified_api':spec['qualified_api'],'source_sha256':spec['source_sha256']}
    contract={'api_name':spec['api_name'],'gap_id':spec['gap_id'],
              'qualified_api':spec['qualified_api'],'source_sha256':spec['source_sha256']}
    artifact.update(artifact_updates or {})
    contract.update(contract_updates or {})
    manifest=write_adapter_handoff(root,
        operator_artifact=artifact, adapter_contract=json.dumps(contract),
        asnumpy_context={path:content},adapter_context_plan=plan,
        skill_files=('references/official.md',),skill_bundle_sha256='b'*64,gap_taskbook=book)
    return root,manifest,book,path


def test_external_handoff_taskbook_files_hashes_and_prompt_match(tmp_path):
    root,manifest,book,path=_handoff(tmp_path)
    ref=manifest['gap_taskbook']
    assert ref['taskbook_id']==book['taskbook_id']
    for key in ('json','markdown'):
        payload=(root/ref[key]['path']).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==ref[key]['sha256']
    assert json.loads((root/ref['json']['path']).read_text())==book
    assert (root/ref['markdown']['path']).read_text() in (root/'prompts'/(path+'.md')).read_text()
    loaded=load_adapter_handoff(root)
    assert loaded['manifest']['gap_taskbook']==ref
    assert loaded['gap_taskbook']==book


@pytest.mark.parametrize('name',['GAP_TASKBOOK.json','GAP_TASKBOOK.md'])
def test_external_handoff_taskbook_byte_drift_is_rejected(tmp_path,name):
    root,manifest,book,path=_handoff(tmp_path)
    target=root/'taskbook'/name
    target.write_text(target.read_text()+'\nchanged\n')
    with pytest.raises(ValueError,match='taskbook hash/path'):
        load_adapter_handoff(root)


def test_external_handoff_rehashed_markdown_drift_is_rejected(tmp_path):
    root,manifest,book,path=_handoff(tmp_path)
    target=root/'taskbook/GAP_TASKBOOK.md'
    target.write_text('forged readable request')
    manifest['gap_taskbook']['markdown']['sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    (root/'handoff.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identity/render'):
        load_adapter_handoff(root)


def test_external_handoff_taskbook_symlink_is_rejected(tmp_path):
    root,manifest,book,path=_handoff(tmp_path)
    target=root/'taskbook/GAP_TASKBOOK.json'
    external=tmp_path/'copy.json';external.write_bytes(target.read_bytes())
    target.unlink();target.symlink_to(external)
    with pytest.raises(ValueError,match='taskbook hash/path'):
        load_adapter_handoff(root)


@pytest.mark.parametrize('location',['artifact','contract'])
@pytest.mark.parametrize('field,bad',[
    ('gap_id','different-gap'),('qualified_api','numpy.sin'),('asnumpy_api','asnumpy.sin'),
    ('api_name','sin'),('source_sha256','0'*64),('gap_source_sha256','1'*64),('gap_spec_sha256','2'*64),
])
def test_other_api_source_or_spec_taskbook_cannot_reach_adapter_callback(tmp_path,monkeypatch,location,field,bad):
    book=_book()
    inputs=_adapter_inputs(tmp_path,book)
    target=inputs['operator_artifact'] if location=='artifact' else json.loads(inputs['adapter_contract'])
    target[field]=bad
    if location=='contract':
        inputs['adapter_contract']=json.dumps(target)
    out=tmp_path/'adapter';out.mkdir()
    provider=LlmDraftProvider(skill_bundle=_FakeSkillBundle())
    monkeypatch.setattr(provider,'_resolve_chat',lambda:pytest.fail('unbound taskbook reached model configuration'))
    with pytest.raises((DraftGenerationError,ValueError),match='taskbook'):
        provider.generate_asnumpy_adapter_candidate(out,**inputs)


def test_handoff_rejects_internally_valid_taskbook_for_other_candidate(tmp_path):
    from numpilot.gaps.taskbook_binding import validate_adapter_taskbook
    book=_book()
    inputs=_adapter_inputs(tmp_path,book)
    inputs['operator_artifact']['gap_id']='another-gap'
    with pytest.raises(ValueError,match='taskbook'):
        validate_adapter_taskbook(book,operator_artifact=inputs['operator_artifact'],adapter_contract=inputs['adapter_contract'])


def test_direct_provider_persists_taskbook_before_model_and_reuse_is_historical(tmp_path):
    from numpilot.gaps.models import GapSpec
    from tests.test_gap_completion import _valid_bundle
    book=_book()
    spec=GapSpec.from_dict(book['gap_spec'])
    workspace=tmp_path/'gap';workspace.mkdir()
    calls=[]
    def chat(_system,prompt):
        persisted=json.loads((workspace/'GAP_TASKBOOK.json').read_text())
        assert persisted['gap_spec']==spec.to_dict()
        assert persisted['taskbook_id'] in prompt
        assert (workspace/'GAP_TASKBOOK.md').read_text() in prompt
        calls.append(1)
        return _valid_bundle()
    provider=LlmDraftProvider(chat=chat,max_attempts=1)
    first=provider.generate(spec,workspace)
    assert len(calls)==1
    (workspace/'GAP_TASKBOOK.json').unlink()
    (workspace/'GAP_TASKBOOK.md').unlink()
    second=provider.generate(spec,workspace)
    assert first.artifact_id==second.artifact_id
    assert len(calls)==1
    assert provider.action_for(second)=='reused_existing'
    assert not (workspace/'GAP_TASKBOOK.json').exists(), 'historical draft reuse must not acquire a fabricated new request'


def test_direct_provider_non_fp32_request_is_saved_and_blocked(tmp_path,monkeypatch):
    from numpilot.gaps.models import GapSpec
    spec=GapSpec.from_dict(_book(dtype='float64')['gap_spec'])
    workspace=tmp_path/'gap';workspace.mkdir()
    provider=LlmDraftProvider()
    monkeypatch.setattr(provider,'_resolve_chat',lambda:pytest.fail('non-fp32 direct request reached model'))
    with pytest.raises(DraftGenerationError,match='float32'):
        provider.generate(spec,workspace)
    saved=json.loads((workspace/'GAP_TASKBOOK.json').read_text())
    assert saved['generation']['allowed'] is False
    assert (workspace/'GAP_TASKBOOK.md').is_file()


def test_direct_provider_preserves_existing_source_verified_taskbook(tmp_path):
    from numpilot.gap_requirements import persist_gap_taskbook
    from numpilot.gaps.models import GapSpec
    from tests.test_gap_completion import _valid_bundle
    book=_book()
    workspace=tmp_path/'gap';workspace.mkdir()
    persist_gap_taskbook(workspace,book)
    def chat(_system,prompt):
        assert json.loads((workspace/'GAP_TASKBOOK.json').read_text())==book
        assert book['taskbook_id'] in prompt
        return _valid_bundle()
    provider=LlmDraftProvider(chat=chat,max_attempts=1)
    provider.generate(GapSpec.from_dict(book['gap_spec']),workspace)
    assert json.loads((workspace/'GAP_TASKBOOK.json').read_text())==book



@pytest.mark.parametrize('location,updates',[
    ('artifact',{'gap_id':'other-gap'}),('contract',{'qualified_api':'numpy.sin'}),
    ('contract',{'gap_spec_sha256':'3'*64}),('artifact',{'source_sha256':'4'*64}),
])
def test_handoff_write_rejects_other_candidate_bindings(tmp_path,location,updates):
    with pytest.raises(ValueError,match='taskbook binding'):
        _handoff(tmp_path,**{location+'_updates':updates})
    assert not (tmp_path/'handoff/prompts').exists()
    assert not (tmp_path/'handoff/responses').exists()


@pytest.mark.parametrize('location,field,bad',[
    ('artifact','gap_id','different-gap'),('contract','qualified_api','numpy.sin'),
    ('contract','gap_spec_sha256','5'*64),('artifact','source_sha256','6'*64),
])
def test_handoff_load_rechecks_binding_after_manifest_rehash(tmp_path,location,field,bad):
    root,manifest,book,path=_handoff(tmp_path)
    if location=='artifact':
        manifest['operator_artifact'][field]=bad
    else:
        contract=json.loads(manifest['adapter_contract']);contract[field]=bad
        manifest['adapter_contract']=json.dumps(contract)
        manifest['adapter_contract_sha256']=hashlib.sha256(manifest['adapter_contract'].encode()).hexdigest()
    (root/'handoff.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='taskbook binding'):
        load_adapter_handoff(root)


def test_non_fp32_handoff_is_saved_as_blocked_without_prompts_or_responses(tmp_path):
    root,manifest,book,path=_handoff(tmp_path,book=_book('matmul',dtype='float64'))
    assert manifest['status']=='requirements_blocked'
    assert (root/'taskbook/GAP_TASKBOOK.json').is_file()
    assert (root/'taskbook/GAP_TASKBOOK.md').is_file()
    assert not (root/'prompts').exists()
    assert not (root/'responses').exists()
    with pytest.raises(ValueError,match='status/protocol'):
        load_adapter_handoff(root)


def test_handoff_taskbook_ref_removal_cannot_downgrade_to_legacy(tmp_path):
    root,manifest,book,path=_handoff(tmp_path)
    manifest.pop('gap_taskbook')
    (root/'handoff.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='reference is missing'):
        load_adapter_handoff(root)


def test_direct_provider_rejects_existing_other_spec_taskbook_before_model(tmp_path,monkeypatch):
    from numpilot.gap_requirements import persist_gap_taskbook
    from numpilot.gaps.models import GapSpec
    book=_book()
    workspace=tmp_path/'gap';workspace.mkdir()
    persist_gap_taskbook(workspace,book)
    other=replace(GapSpec.from_dict(book['gap_spec']),target_soc='Ascend910B4')
    provider=LlmDraftProvider()
    monkeypatch.setattr(provider,'_resolve_chat',lambda:pytest.fail('wrong source/spec reached model'))
    with pytest.raises(DraftGenerationError,match='taskbook'):
        provider.generate(other,workspace)
    assert json.loads((workspace/'GAP_TASKBOOK.json').read_text())==book
