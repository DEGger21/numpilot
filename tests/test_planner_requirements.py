"""Requirements remain inspectable before any model or runtime exists."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.gap_filler import GapFillerAgent
from numpilot.agents.planner import PlannerAgent, MigrationPlan, ROUTE_DIRECT
from numpilot.contracts import AnalysisResult, OpUse
from numpilot.gap_requirements import (
    build_gap_taskbook, build_planning_taskbook, persist_gap_taskbook,
    render_gap_taskbook_markdown, validate_gap_taskbook,
)
from numpilot.gaps.pipeline import (
    GapCompletionError, GapCompletionPipeline, prepare_gap_completion,
    persist_preparation_taskbooks,
)
from numpilot.gaps.provider import DraftGenerationError, _compact_generation_contract
from numpilot.gaps.spec import build_gap_specs, render_generation_request

SOURCE = '''import numpy as np

def forward(x):
    return np.cbrt(x)
'''


def _spec(source=SOURCE):
    analysis = AnalysisAgent().analyze(source, 'case.py')
    return build_gap_specs(source, analysis, target_soc='Ascend910B2', cann_version='9.2')[0]


def _reseal(value):
    value.pop('taskbook_id', None)
    digest = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    value['taskbook_id'] = 'gap-taskbook-' + digest
    return value


def test_canonical_taskbook_is_deterministic_and_not_a_qualification():
    spec = _spec()
    value = build_gap_taskbook(spec, source=SOURCE)
    assert value == build_gap_taskbook(spec, source=SOURCE)
    assert value['gap_spec'] == spec.to_dict()
    assert value['source']['sha256'] == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert value['call_contract']['observed_profiles'] == []
    assert all(item['status'] == 'unknown' for item in value['call_contract']['tensor_facts'].values())
    assert value['support_scope']['input_dtype'] == 'float32'
    assert value['support_scope']['dtype_source'] == 'project_policy_not_observed_profile'
    assert value['acceptance']['status'] == 'pending'
    assert value['acceptance']['planner_self_certification'] is False
    assert value['responsibility']['generation_owner'] == 'gap_filler'
    assert value['responsibility']['verification_owner'] == 'deterministic_validator'
    assert value['generation']['allowed'] is True


def test_static_arguments_preserve_axes_literals_without_inventing_tensor_facts():
    source = 'import numpy as np\ndef forward(x):\n    return np.new_op(x, axis=-2, keepdims=True, size=(3, 4))\n'
    value = build_gap_taskbook(_spec(source), source=source)
    args = value['call_contract']['callsites'][0]['arguments']
    assert args['positional'][0]['status'] == 'unknown'
    assert args['positional'][0]['expression'] == 'x'
    keywords = {item['name']: item for item in args['keywords']}
    assert keywords['axis']['value'] == -2
    assert keywords['keepdims']['value'] is True
    assert keywords['size']['value'] == [3, 4]
    assert value['call_contract']['tensor_facts']['input_shape']['status'] == 'unknown'


def test_source_is_never_executed_and_source_hash_is_checked(tmp_path):
    marker = tmp_path/'executed'
    source = 'open(%r, "w").write("bad")\n' % str(marker) + SOURCE
    value = build_gap_taskbook(_spec(source), source=source)
    assert value['source']['content_checked'] is True
    assert not marker.exists()
    with pytest.raises(ValueError, match='SHA-256'):
        build_gap_taskbook(_spec(), source=SOURCE+'# changed\n')


def test_callsite_must_exist_in_bound_source():
    spec = _spec()
    spec = replace(spec, callsites=(replace(spec.callsites[0], lineno=2),))
    with pytest.raises(ValueError, match='callsite'):
        build_gap_taskbook(spec, source=SOURCE)


def test_unknown_and_truncated_arguments_stay_unknown():
    spec = _spec()
    spec = replace(spec, callsites=(replace(spec.callsites[0], call_expr='np.cbrt('),))
    value = build_gap_taskbook(spec)
    assert value['source']['content_checked'] is False
    assert value['call_contract']['callsites'][0]['arguments']['status'] == 'unknown'


def test_fp32_policy_rejects_explicit_other_dtype_without_rewriting_spec():
    spec = replace(_spec(), dtype_rules=('float64',))
    value = build_gap_taskbook(spec, source=SOURCE)
    assert value['generation']['allowed'] is False
    assert value['status'] == 'requirements_blocked'
    assert value['gap_spec']['dtype_rules'] == ['float64']
    assert value['support_scope']['input_dtype'] == 'float32'


def test_fp32_is_not_a_claim_that_comparison_output_is_float():
    value = build_gap_taskbook(_spec())
    assert 'bool' in value['support_scope']['output_dtype']
    assert value['call_contract']['tensor_facts']['output_dtype']['status'] == 'unknown'


def test_numpy_oracle_is_executable_but_has_no_fabricated_inputs():
    import numpy as np
    value = build_gap_taskbook(_spec())
    namespace = {}
    exec(value['acceptance']['oracle']['python'], namespace)
    inputs = np.array([-8., -0., 0., 27.], dtype=np.float32)
    actual = namespace['numpy_oracle'](inputs)
    np.testing.assert_array_equal(actual, np.cbrt(inputs))
    assert value['acceptance']['oracle']['status'] == 'inputs_required'
    assert value['acceptance']['tolerance']['status'] == 'unknown'
    assert {gate['id'] for gate in value['acceptance']['gates']} >= {'numerical', 'whole_program', 'native_execution', 'performance'}


def test_draft_plan_does_not_construct_fake_gap_spec_or_environment():
    analysis = AnalysisAgent().analyze(SOURCE, 'case.py')
    gaps = GapFillerAgent().plan(analysis)
    plan = PlannerAgent().plan(analysis, gaps)
    value = build_planning_taskbook(analysis, gaps, source=SOURCE)
    assert plan.to_dict()['planning_requirements'][0]['callsites'] == value['requirements'][0]['callsites']
    assert 'source SHA-256' not in value['requirements'][0]['unresolved']
    assert value['generation']['allowed'] is False
    assert value['status'] == 'requirements_incomplete'
    assert 'gap_spec' not in value
    assert value['integration']['target_soc'] is None
    assert value['integration']['cann_version'] is None
    assert value['requirements'][0]['callsites'][0]['qualified_api'] == 'numpy.cbrt'
    assert 'canonical GapSpec' in value['unresolved']


def test_keyword_and_lowering_gaps_do_not_become_kernel_generation():
    analysis = AnalysisResult('case.py', [OpUse('add', 4, False, reason='AsNumpy 未验证 NumPy 关键字: out'),
        OpUse('python_operator.Lt', 5, False)], ['add','python_operator.Lt'], False)
    plan = PlannerAgent().plan(analysis, GapFillerAgent().plan(analysis))
    assert {row['next_role'] for row in plan.planning_requirements} == {'consent_gate', 'code_migrator'}
    assert all(row['status'] == 'requirements_incomplete' for row in plan.planning_requirements)


def test_direct_and_legacy_plan_construction_remain_compatible():
    legacy = MigrationPlan(ROUTE_DIRECT, {'sin':'NPU'}, 1, 'whole_forward_npu')
    assert legacy.planning_requirements == ()
    analysis = AnalysisResult('case.py', [OpUse('sin', 4, True)], [], True)
    assert PlannerAgent().plan(analysis, GapFillerAgent().plan(analysis)).planning_requirements == ()


def test_unexplained_analysis_failure_is_explicitly_incomplete():
    analysis = AnalysisResult('case.py', [], [], False)
    value = build_planning_taskbook(analysis, GapFillerAgent().plan(analysis))
    assert value['generation']['allowed'] is False
    assert any('diagnostics' in item for item in value['unresolved'])


def test_json_markdown_written_once_idempotently_and_hash_bound(tmp_path):
    value = build_gap_taskbook(_spec(), source=SOURCE)
    ref = persist_gap_taskbook(tmp_path, value)
    assert persist_gap_taskbook(tmp_path, value) == ref
    for key in ('json','markdown'):
        payload = Path(ref[key]['path']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == ref[key]['sha256']
    assert json.loads(Path(ref['json']['path']).read_text()) == value
    text = Path(ref['markdown']['path']).read_text()
    assert '#/gap_spec/...' in text
    assert 'float32' in text and 'bool' in text and value['taskbook_id'] in text
    assert text == render_gap_taskbook_markdown(value)
    Path(ref['markdown']['path']).write_text('changed')
    with pytest.raises(ValueError, match='different taskbook'):
        persist_gap_taskbook(tmp_path, value)


def test_taskbook_tamper_and_unknown_schema_are_rejected(tmp_path):
    value = build_gap_taskbook(_spec())
    value['support_scope']['input_dtype'] = 'float64'
    with pytest.raises(ValueError, match='identity'):
        persist_gap_taskbook(tmp_path, value)
    _reseal(value)
    with pytest.raises(ValueError, match='float32 request'):
        validate_gap_taskbook(value)
    value = build_gap_taskbook(_spec())
    value['schema_version'] = True
    with pytest.raises(ValueError, match='schema'):
        validate_gap_taskbook(_reseal(value))


def test_gap_spec_source_and_hash_are_not_relabelable():
    value = build_gap_taskbook(_spec())
    value['gap_spec']['dtype_rules'] = ['float64']
    with pytest.raises(ValueError, match='binding'):
        validate_gap_taskbook(_reseal(value))


def test_taskbook_never_follows_symlink(tmp_path):
    output=tmp_path/'out'; output.mkdir()
    victim=tmp_path/'victim'; victim.write_text('untouched')
    (output/'GAP_TASKBOOK.json').symlink_to(victim)
    with pytest.raises((ValueError,OSError)):
        persist_gap_taskbook(output, build_gap_taskbook(_spec()))
    assert victim.read_text() == 'untouched'
    assert not (output/'GAP_TASKBOOK.md').exists()


def test_same_taskbook_is_in_model_and_compacted_stage_request():
    spec = _spec()
    value = build_gap_taskbook(spec, source=SOURCE)
    prompt = render_generation_request(spec, taskbook=value)
    compact = _compact_generation_contract(prompt+'\n## Embedded official Huawei CANNBot skill\nlarge skill')
    assert value['taskbook_id'] in compact
    assert 'Machine-readable taskbook' in compact
    assert json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) in compact
    with pytest.raises(ValueError, match='does not match'):
        render_generation_request(replace(spec, target_soc='other'), taskbook=value)


def test_pipeline_persists_all_taskbooks_before_provider_failure(tmp_path):
    source=tmp_path/'program.py'; source.write_text('import numpy as np\ndef forward(x):\n    return np.new_one(np.new_two(x))\n')
    out=tmp_path/'out'
    observed=[]
    class Provider:
        def generate(self,spec,workspace,*,generation_request):
            books = list((out/'gaps').glob('*/GAP_TASKBOOK.json'))
            assert len(books) == 2
            book=json.loads((workspace/'GAP_TASKBOOK.json').read_text())
            assert book['taskbook_id'] in generation_request
            observed.append(spec.gap_id)
            raise DraftGenerationError('test provider stopped')
    with pytest.raises(GapCompletionError, match='draft generation failed'):
        GapCompletionPipeline(provider=Provider()).run(str(source),str(out), target_soc='Ascend910B2',cann_version='9.2')
    assert len(observed) == 1
    assert len(list((out/'gaps').glob('*/GAP_TASKBOOK.md'))) == 2


def test_preparation_taskbooks_available_without_provider_or_model_config(tmp_path):
    source=tmp_path/'program.py'; source.write_text(SOURCE)
    prep=prepare_gap_completion(str(source),target_soc='Ascend910B2',cann_version='9.2')
    refs=persist_preparation_taskbooks(prep,str(tmp_path/'out'))
    assert len(refs)==1 and Path(refs[0]['json']['path']).is_file()
    result=GapCompletionPipeline().run(str(source),str(tmp_path/'out'),target_soc='Ascend910B2',cann_version='9.2',preparation=prep)
    assert result.to_dict()['taskbooks'] == list(refs)
    assert result.generated_count == 0


def test_explicit_non_fp32_operator_is_saved_but_never_sent_to_provider(tmp_path):
    source = tmp_path / 'domain.py'
    source.write_text("""import numpy as np
from numpilot import operator_spec
@operator_spec(name='domain_fp64',version='1',numpy_oracle='oracle',target_api='asnumpy.domain_fp64',
 provenance='custom_generation_candidate',route='ascendc_registry_candidate',
 signature_variants=({'positional_args':1,'keyword_args':()},),dtype_rules=('float64',),
 shape_rules=('rank>=1',),source='local custom NumPy oracle',license='Apache-2.0',
 target_soc='Ascend910B2',cann_version='9.2')
def oracle(x):
    return np.cbrt(x)
def forward(x):
    return oracle(x)
""")
    class ForbiddenProvider:
        def generate(self,*args,**kwargs):
            raise AssertionError('non-float32 requirement reached model')
    output=tmp_path/'out'
    with pytest.raises(GapCompletionError,match='float32 generation scope'):
        GapCompletionPipeline(provider=ForbiddenProvider()).run(str(source),str(output),target_soc='Ascend910B2',cann_version='9.2')
    books=list((output/'gaps').glob('*/GAP_TASKBOOK.json'))
    assert len(books)==1
    assert json.loads(books[0].read_text())['generation']['allowed'] is False
    assert (books[0].parent/'GAP_TASKBOOK.md').is_file()
    assert not (books[0].parent/'draft').exists()


def test_rehashed_non_fp32_taskbook_cannot_relabel_generation_allowed():
    value = build_gap_taskbook(replace(_spec(), dtype_rules=('float64',)))
    value['generation']={'allowed':True,'blockers':[],'claim_level':'unverified_candidate_only'}
    value['status']='requirements_prepared_unverified'
    with pytest.raises(ValueError,match='generation scope'):
        validate_gap_taskbook(_reseal(value))


def test_no_gaps_preparation_does_not_create_empty_taskbook_directory(tmp_path):
    source=tmp_path/'program.py';source.write_text('import numpy as np\ndef forward(x):\n    return np.sin(x)\n')
    prep=prepare_gap_completion(str(source),target_soc='Ascend910B2',cann_version='9.2')
    assert not prep.specs
    output=tmp_path/'untouched'
    assert persist_preparation_taskbooks(prep,str(output)) == ()
    assert not output.exists()


def test_taskbook_preparation_rejects_parent_symlink_before_any_write(tmp_path):
    source=tmp_path/'program.py';source.write_text(SOURCE)
    prep=prepare_gap_completion(str(source),target_soc='Ascend910B2',cann_version='9.2')
    real=tmp_path/'real';real.mkdir()
    linked=tmp_path/'linked';linked.symlink_to(real,target_is_directory=True)
    with pytest.raises(GapCompletionError,match='not a real directory'):
        persist_preparation_taskbooks(prep,str(linked/'new-output'))
    assert list(real.iterdir()) == []
