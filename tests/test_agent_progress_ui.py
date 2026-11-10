"""CPU-only terminal/event wiring; no model or NPU is used by these tests."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from numpilot import cli_ui
from numpilot.progress import ProgressEmitter, RoleProgressEvent
from numpilot.roles import ROLE_ORDER, Role


class FlushedStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0
    def flush(self):
        self.flushes += 1
        return super().flush()


@pytest.fixture(autouse=True)
def no_transport_fd(monkeypatch):
    monkeypatch.delenv('NUMPILOT_PROGRESS_FD', raising=False)


def test_plain_events_are_immediate_ordered_and_keep_six_canonical_roles():
    stream = FlushedStream()
    with cli_ui.AgentProgress(stream=stream) as view:
        assert list(view.rows) == [role.value for role in ROLE_ORDER]
        view.emit('analyzer', 'running', '解析代码')
        assert '执行中' in stream.getvalue() and stream.flushes >= 1
        view.emit('analyzer', 'completed', '解析完成')
        view.emit(None, 'running', '验证数值结果', component='verification')
        view.emit(None, 'failed', '数值验证失败', component='verification')
        assert len(view.rows) == 6
        assert view.rows['code_migrator']['state'] == 'pending'
    output = stream.getvalue()
    assert output.index('解析代码') < output.index('解析完成') < output.index('数值验证失败')
    assert '[验证步骤]' in output and '规则处理 · 无模型' in output
    assert '真实调用' not in output and '\x1b' not in output
    assert view.value == 'error'


def test_execution_modes_follow_events_not_engine_assumptions():
    stream = io.StringIO()
    with cli_ui.AgentProgress(stream=stream) as view:
        view.emit('planner', 'completed', '规则规划完成')
        view.emit('code_migrator', 'running', '等待迁移代码', execution_mode='llm')
        view.emit('code_migrator', 'completed', '已取得迁移代码', execution_mode='llm')
        view.emit('gap_filler', 'blocked', '等待外部 Codex 接管', execution_mode='external')
    lines = stream.getvalue().splitlines()
    assert '无模型' in lines[0] and '真实调用' not in lines[0]
    assert '真实调用' in lines[1]
    assert '外部代理' in lines[-1] and '已阻断' in lines[-1]
    assert view.rows['diagnostics']['state'] == 'pending'
    assert view.value == 'warn'


def test_elapsed_uses_monotonic_clock_and_freezes_terminal_state(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(cli_ui.time, 'monotonic', lambda: clock[0])
    stream = io.StringIO()
    with cli_ui.AgentProgress(stream=stream) as view:
        view.emit('analyzer', 'running', '处理中')
        clock[0] = 103.25
        assert view._elapsed(view.rows['analyzer']) == 3.25
        view.emit('analyzer', 'completed', '处理完成')
        clock[0] = 120.0
        assert view._elapsed(view.rows['analyzer']) == 3.25
    assert '3.2s' in stream.getvalue()


def test_exception_marks_active_role_failed_without_printing_exception_secret():
    stream = io.StringIO()
    with pytest.raises(RuntimeError, match='provider-private-token'):
        with cli_ui.AgentProgress(stream=stream) as view:
            view.emit('code_migrator', 'running', '模型正在生成', execution_mode='llm')
            raise RuntimeError('provider-private-token')
    assert view.rows['code_migrator']['state'] == 'failed'
    assert '失败' in stream.getvalue()
    assert 'provider-private-token' not in stream.getvalue()


def test_early_return_does_not_claim_running_or_pending_roles_completed():
    stream = io.StringIO()
    with cli_ui.AgentProgress(stream=stream) as view:
        view.emit('planner', 'running', '核对资格')
        view.warning()
    assert view.rows['planner']['state'] == 'blocked'
    assert view.rows['reporter']['state'] == 'pending'
    assert '已阻断' in stream.getvalue()


def test_disabled_ui_keeps_stdout_and_stderr_clean(capsys):
    with cli_ui.AgentProgress(enabled=False) as view:
        view.emit('analyzer', 'completed', '扫描完成')
    captured = capsys.readouterr()
    assert captured.out == captured.err == ''
    assert view.rows['analyzer']['state'] == 'completed'


def test_default_stream_is_stderr_only(capsys):
    with cli_ui.AgentProgress() as view:
        view.emit('analyzer', 'completed', '扫描完成')
    captured = capsys.readouterr()
    assert captured.out == '' and '扫描完成' in captured.err


@pytest.mark.parametrize('message,hidden', [
    ('api_key=private-key 连接失败', 'private-key'),
    ('Authorization: Bearer private-access-token', 'private-access-token'),
    ('auth_token="private-token"', 'private-token'),
    ('password: private-password', 'private-password'),
    ('调用 sk-privateabcdefghij 等待', 'sk-privateabcdefghij'),
])
def test_progress_redacts_credential_like_labels(message, hidden):
    stream = io.StringIO()
    with cli_ui.AgentProgress(stream=stream) as view:
        view.emit('gap_filler', 'blocked', message)
    assert hidden not in stream.getvalue()
    assert '隐藏' in stream.getvalue()


def test_progress_labels_are_bounded_and_control_characters_never_pass():
    text = cli_ui._progress_message('a' * 300 + '\x1b[2J\nprovider')
    assert len(text) <= 181 and '\x1b' not in text and '\n' not in text


@pytest.mark.parametrize('forwarded', [True, False])
def test_child_event_transport_precedes_local_rendering(monkeypatch, forwarded):
    received = []
    def emit_child_event(event):
        received.append(event)
        return forwarded
    monkeypatch.setitem(sys.modules, 'numpilot.progress_transport',
                        SimpleNamespace(emit_child_event=emit_child_event))
    stream = io.StringIO()
    with cli_ui.AgentProgress(stream=stream) as view:
        view.emit('reporter', 'completed', '报告已保存')
    assert len(received) == 1 and received[0].role is Role.REPORTER
    assert view.rows['reporter']['state'] == 'completed'
    assert ('报告已保存' in stream.getvalue()) is (not forwarded)


def test_strict_dict_event_rejects_extra_provider_data_without_rendering():
    stream = io.StringIO()
    event = RoleProgressEvent(Role.ANALYZER, 'analyzer', 'running', '分析代码').to_dict()
    event['provider_response'] = 'private-provider-message'
    with cli_ui.AgentProgress(stream=stream) as view:
        with pytest.raises(ValueError):
            view.on_event(event)
    assert stream.getvalue() == ''


def test_optional_rich_absence_falls_back_to_plain(monkeypatch):
    monkeypatch.setattr(cli_ui, '_RICH_AVAILABLE', False)
    stream = io.StringIO()
    with cli_ui.AgentProgress(stream=stream) as view:
        view.emit('analyzer', 'completed', '规则分析完成')
    assert '规则分析完成' in stream.getvalue() and '\x1b' not in stream.getvalue()


@pytest.mark.skipif(not cli_ui.rich_available(), reason='optional rich not installed')
def test_tty_view_has_six_roles_and_separate_verification_without_markup(monkeypatch):
    from rich.console import Console
    captured = io.StringIO()
    with cli_ui.AgentProgress(stream=io.StringIO()) as view:
        view.emit('gap_filler', 'blocked', '[bold red]literal label[/]', execution_mode='external')
        view.emit(None, 'completed', '独立验证步骤', component='verification')
        panel = view._renderable()
        Console(file=captured, width=240, color_system=None).print(panel)
    output = captured.getvalue()
    for role in ('Analyzer', 'Planner', 'Gap Filler', 'Migrator', 'Diagnostics', 'Reporter'):
        assert role in output
    assert '验证步骤' in output and '六角色协作' in output
    assert '[bold red]literal label[/]' in output
    assert '运行状态不替代验证回执' in output


def test_cli_run_json_stdout_is_parseable_with_events(tmp_path, monkeypatch, capsys):
    from numpilot import __main__ as cli
    from numpilot import workflow
    received = []
    def fake_workflow(source, out, *, on_event, **kwargs):
        emitter = ProgressEmitter(on_event)
        emitter.emit('analyzer', 'completed', '静态扫描完成')
        emitter.emit(None, 'completed', '验证结束', component='verification')
        received.append(on_event)
        return {'status': 'passed', 'fixture': 'CPU observer test'}
    monkeypatch.setattr(workflow, 'run_ordinary_workflow', fake_workflow)
    result = cli.main(['run', str(tmp_path / 'fixture.py'), '--out', str(tmp_path / 'run'), '--json'])
    captured = capsys.readouterr()
    assert result == 0 and len(received) == 1
    assert json.loads(captured.out) == {'status': 'passed', 'fixture': 'CPU observer test'}
    assert captured.err == ''


def test_contract_cli_ast_shim_emits_real_role_events_without_model(tmp_path):
    source = tmp_path / 'source.py'
    source.write_text('import numpy as np\n'
        'def build_inputs():\n    return {"x": np.arange(8, dtype=np.float32)}\n'
        'def forward(x):\n    return np.sin(x)\n')
    environment = os.environ.copy()
    environment['NUMPILOT_PLAIN'] = '1'
    environment.pop('NUMPILOT_PROGRESS_FD', None)
    result = subprocess.run([sys.executable, '-m', 'numpilot', 'migrate', str(source),
        '--mode', 'contract', '--engine', 'ast', '--backend', 'shim', '--skip-env-check',
        '--out', str(tmp_path / 'out'), '--reports-dir', str(tmp_path / 'reports')],
        text=True, capture_output=True, env=environment, timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'Analyzer' in result.stderr and 'Reporter' in result.stderr
    assert '执行中' in result.stderr and '验证步骤' in result.stderr
    assert '规则处理 · 无模型' in result.stderr and '真实调用' not in result.stderr
    assert '[报告]' in result.stdout and '\x1b' not in result.stderr


@pytest.mark.skipif(not cli_ui.rich_available(), reason='optional rich not installed')
def test_real_tty_live_context_starts_and_stops_cleanly(monkeypatch):
    class TerminalBuffer(io.StringIO):
        def isatty(self):
            return True
    for name in ('NO_COLOR', 'NUMPILOT_PLAIN'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('TERM', 'xterm-256color')
    stream = TerminalBuffer()
    with cli_ui.AgentProgress(stream=stream) as view:
        assert view._live is not None and view._live.is_started
        view.emit('analyzer', 'running', '处理真实事件')
        view.emit('analyzer', 'completed', '分析结束')
    assert not view._live.is_started
    assert '六角色协作' in stream.getvalue() and '分析结束' in stream.getvalue()


def test_complete_gap_taskbook_survives_block_before_llm_access(tmp_path, monkeypatch, capsys):
    from numpilot import __main__ as cli
    source = tmp_path / 'gap.py'
    source.write_text('import numpy as np\n'
        'def build_inputs():\n    return {"x": np.ones(8, dtype=np.float32)}\n'
        'def forward(x):\n    return np.cbrt(x)\n')
    out = tmp_path / 'blocked-gap'
    def forbidden(*args, **kwargs):
        raise AssertionError('model access before taskbook/official admission')
    monkeypatch.setattr(cli, '_load_llm_or_report', forbidden)
    monkeypatch.setattr(cli, '_run_llm_readiness_gate', forbidden)
    assert cli.main(['complete-gaps', str(source), '--provider', 'llm', '--out', str(out),
                     '--target-soc', 'Ascend910B4', '--cann-version', '9.0.0']) == 3
    books = list(out.glob('gaps/*/GAP_TASKBOOK.json'))
    assert len(books) == 1
    taskbook = json.loads(books[0].read_text())
    assert taskbook['gap_spec']['qualified_api'] == 'numpy.cbrt'
    assert taskbook['generation']['claim_level'] == 'unverified_candidate_only'
    assert {path.name for path in out.rglob('*') if path.is_file()} == {'GAP_TASKBOOK.json', 'GAP_TASKBOOK.md'}
    assert '[任务书]' in capsys.readouterr().err


def test_explicit_non_float32_gap_saves_requirements_and_blocks_generation(tmp_path, monkeypatch, capsys):
    from numpilot import __main__ as cli
    source = tmp_path / 'gap.py'
    source.write_text('import numpy as np\n'
        'def build_inputs():\n    return {"x": np.ones(8, dtype=np.float32)}\n'
        'def forward(x):\n    return np.cbrt(x, dtype=np.float64)\n')
    out = tmp_path / 'blocked-dtype'
    monkeypatch.setattr(cli, '_load_llm_or_report', lambda: pytest.fail('must not load model'))
    assert cli.main(['complete-gaps', str(source), '--provider', 'llm', '--out', str(out),
                     '--target-soc', 'Ascend910B4', '--cann-version', '9.0.0']) == 2
    taskbook = json.loads(next(out.glob('gaps/*/GAP_TASKBOOK.json')).read_text())
    assert taskbook['generation']['allowed'] is False
    assert taskbook['status'] == 'requirements_blocked'
    assert '首期 float32' in capsys.readouterr().err


def test_adapter_taskbook_is_durable_before_configuration_is_read(tmp_path, monkeypatch, capsys):
    import hashlib
    from numpilot import __main__ as cli
    from numpilot.agents.analysis import AnalysisAgent
    from numpilot.gaps.spec import build_gap_specs
    from numpilot.gaps import provider
    source = tmp_path / 'gap.py'
    source.write_text('import numpy as np\n'
        'def build_inputs():\n    return {"x": np.ones(8, dtype=np.float32)}\n'
        'def forward(x):\n    return np.cbrt(x)\n')
    text = source.read_text()
    spec = build_gap_specs(text, AnalysisAgent().analyze(text, str(source)),
                          target_soc='ascend910b', cann_version='9.0.0')[0]
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(spec.to_dict()))
    artifact = tmp_path / 'artifact'
    adapter = artifact / 'asnumpy_adapter/operator.json'
    adapter.parent.mkdir(parents=True)
    adapter.write_text('{"api_name":"cbrt"}\n')
    (artifact / 'operator-artifact.json').write_text(json.dumps({'artifact_id': 'draft-' + 'a' * 20}))
    signed = SimpleNamespace(files={'asnumpy_adapter/operator.json': hashlib.sha256(adapter.read_bytes()).hexdigest()})
    monkeypatch.setattr(provider, '_load_existing_draft', lambda *args: signed)
    reference = tmp_path / 'asnumpy-source'
    for name in ('include/asnumpy/nn/activation.hpp', 'csrc/nn/activation.cpp',
                 'bindings/python/bind_nn.cpp', 'bindings/python/module.cpp',
                 'src/asnumpy/nn.py', 'src/asnumpy/__init__.py',
                 'bindings/python/CMakeLists.txt', 'tests/asnumpy_tests/nn_tests/test_activation.py'):
        path = reference / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# CPU fixture\n')
    out = tmp_path / 'adapter-out'
    calls = []
    def check_saved_first():
        book = tmp_path / 'adapter-out-taskbook/GAP_TASKBOOK.json'
        assert book.is_file()
        value = json.loads(book.read_text())
        assert value['gap_spec_sha256'] and value['gap_spec']['gap_id'] == spec.gap_id
        assert not out.exists()
        calls.append('configuration')
        return None
    monkeypatch.setattr(cli, '_load_llm_or_report', check_saved_first)
    monkeypatch.setattr(cli, '_run_llm_readiness_gate', lambda *args, **kwargs: pytest.fail('no configuration'))
    assert cli.main(['generate-asnumpy-adapter', '--spec', str(spec_path), '--artifact', str(artifact),
                     '--asnumpy-reference', str(reference), '--out', str(out)]) == 4
    assert calls == ['configuration'] and not out.exists()
    assert (tmp_path / 'adapter-out-taskbook/GAP_TASKBOOK.md').is_file()
    assert '[任务书]' in capsys.readouterr().err


def test_wizard_default_runs_do_not_reuse_immutable_taskbook_directory(tmp_path, monkeypatch):
    from numpilot import wizard
    root = tmp_path / 'wizard-runs'
    monkeypatch.setattr(wizard, 'default_wizard_dir', lambda: root)
    monkeypatch.setattr(wizard, '_startup_check', lambda backend, interactive: 'shim')
    for _ in range(2):
        assert wizard.run_wizard(file='examples/unsupported_demo.py', engine='ast', backend='shim',
                                 reports_dir=str(tmp_path / 'reports')) == 1
    runs = list(root.glob('run-*'))
    assert len(runs) == 2 and all(path.is_dir() for path in runs)
    assert all((path / 'GAP_TASKBOOK.json').is_file() for path in runs)


def test_standalone_analyze_json_forwards_analyzer_events_without_source_execution(tmp_path, monkeypatch, capsys):
    from numpilot import __main__ as cli
    from numpilot.backends.npu import NpuBackend
    source = tmp_path / 'source.py'
    source.write_text('import numpy as np\nx=np.full((2,),1.0,dtype=np.float32)\ny=np.exp(x)\n')
    received = []
    monkeypatch.setitem(sys.modules, 'numpilot.progress_transport',
                        SimpleNamespace(emit_child_event=lambda event: received.append(event) or True))
    monkeypatch.setattr(NpuBackend, 'runtime_identity', lambda self: SimpleNamespace(
        status='matched', identity_sha256='a' * 64, audit_id='test-runtime-audit-v1'))
    monkeypatch.setattr(NpuBackend, 'get_module', lambda self: pytest.fail('analyze must not load AsNumpy'))
    monkeypatch.setattr(cli, '_load_llm_or_report', lambda: pytest.fail('analyze must not read model config'))
    assert cli.main(['analyze', str(source), '--json']) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload['decision'] == 'DIRECT' and captured.err == ''
    assert [(event.role, event.state) for event in received] == [
        (Role.ANALYZER, 'running'), (Role.ANALYZER, 'completed')]
    assert all(event.execution_mode == 'static' for event in received)


def test_standalone_analyze_failure_closes_analyzer_without_claiming_completion(tmp_path, monkeypatch, capsys):
    from numpilot import __main__ as cli
    received = []
    monkeypatch.setitem(sys.modules, 'numpilot.progress_transport',
                        SimpleNamespace(emit_child_event=lambda event: received.append(event) or True))
    assert cli.main(['analyze', str(tmp_path / 'missing.py'), '--json']) == 2
    assert [(event.role, event.state) for event in received] == [
        (Role.ANALYZER, 'running'), (Role.ANALYZER, 'failed')]
    assert capsys.readouterr().out == ''


def test_bench_cpu_boundary_events_do_not_start_background_live_refresh(tmp_path, monkeypatch):
    from numpilot import __main__ as cli
    received, views, calls = [], [], []
    monkeypatch.setitem(sys.modules, 'numpilot.progress_transport',
                        SimpleNamespace(emit_child_event=lambda event: received.append(event) or True))
    monkeypatch.setattr(cli_ui, 'is_interactive', lambda stream=None, environment=None: True)
    class TrackingView(cli_ui.AgentProgress):
        def __enter__(self):
            result = super().__enter__()
            views.append(self)
            return result
    monkeypatch.setattr(cli, 'AgentProgress', TrackingView)
    def baseline(file, out, threads):
        assert len(received) == 1 and received[0].state == 'running'
        assert views[0].dynamic is False and views[0]._live is None
        calls.append((file, out, threads))
        return 0
    monkeypatch.setattr(cli, 'run_cpu_baseline', baseline)
    source, out = tmp_path / 'source.py', tmp_path / 'baseline'
    assert cli.main(['bench-cpu', str(source), '--out', str(out), '--cpu-threads', '1']) == 0
    assert calls == [(str(source), str(out), cli.parse_cpu_threads('1'))]
    assert [(event.role, event.component, event.state) for event in received] == [
        (None, 'verification', 'running'), (None, 'verification', 'completed')]
    assert 'CPU 基线' in received[0].message


@pytest.mark.parametrize('code,state', [(0, 'completed'), (1, 'blocked'), (2, 'failed')])
def test_bench_npu_wrapper_preserves_return_code_arguments_and_only_system_events(tmp_path, monkeypatch, code, state):
    from numpilot import __main__ as cli
    received, calls = [], []
    backend, profiles = object(), object()
    engine = SimpleNamespace(name='ast')
    monkeypatch.setitem(sys.modules, 'numpilot.progress_transport',
                        SimpleNamespace(emit_child_event=lambda event: received.append(event) or True))
    monkeypatch.setattr(cli, '_qualified_runtime_inputs', lambda args, name: (backend, profiles))
    monkeypatch.setattr(cli, '_preflight', lambda name, skip: None)
    monkeypatch.setattr(cli, '_make_engine', lambda name: engine)
    def benchmark(file, out, **kwargs):
        assert len(received) == 1 and received[0].state == 'running'
        assert kwargs == {'engine': engine, 'threshold': 123, 'allow_cpu_fallback': False,
                          'backend': backend, 'analysis_profiles': profiles}
        calls.append((file, out))
        return code
    monkeypatch.setattr(cli, 'run_npu_compare', benchmark)
    source, out = tmp_path / 'source.py', tmp_path / 'benchmark'
    assert cli.main(['bench-npu', str(source), '--out', str(out), '--engine', 'ast',
                     '--threshold', '123', '--skip-env-check']) == code
    assert calls == [(str(source), str(out))]
    assert [(event.role, event.component, event.state) for event in received] == [
        (None, 'workflow', 'running'), (None, 'workflow', state)]
    assert '严格 NPU 性能' in received[0].message
