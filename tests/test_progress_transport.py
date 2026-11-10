"""Real CPU child-process progress transport; never a model/NPU call."""
import errno
import json
import os
from pathlib import Path
import sys

import pytest

from numpilot import progress_transport as transport
from numpilot import workflow
from numpilot.progress import ProgressEmitter, RoleProgressEvent
from numpilot.roles import Role

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != 'posix', reason='dedicated inherited pipe is POSIX-only')


def _event(**changes):
    fields = dict(role=Role.CODE_MIGRATOR, component='code_migrator', state='running',
                  message='处理代码', execution_mode='static')
    fields.update(changes)
    return RoleProgressEvent(**fields)


def _stage(tmp_path, script, observer, *, timeout=10):
    return workflow._run_stage('test', [sys.executable, '-c', script], tmp_path,
                               cwd=REPO, timeout_seconds=timeout, on_event=observer)


def _ack_wait_script(ack):
    return "\n".join([
        'import time',
        'from pathlib import Path',
        'ack = Path(' + repr(str(ack)) + ')',
        'deadline = time.monotonic() + 8',
        'while not ack.exists():',
        '    if time.monotonic() > deadline: raise SystemExit(91)',
        '    time.sleep(0.01)',
    ])


def test_actual_child_event_arrives_before_child_can_exit(tmp_path):
    ack = tmp_path / 'parent-received.ack'
    finished = tmp_path / 'child-finished'
    received = []
    def observer(event):
        assert not finished.exists(), 'event arrived only after child completion'
        received.append(event)
        ack.write_text('parent observed running event')
    script = '\n'.join([
        'from numpilot.progress import RoleProgressEvent',
        'from numpilot.progress_transport import emit_child_event',
        "event = RoleProgressEvent('code_migrator', 'code_migrator', 'running', '子进程正在执行')",
        'assert emit_child_event(event)',
        _ack_wait_script(ack),
        'Path(' + repr(str(finished)) + ').write_text("finished after acknowledgment")',
    ])
    result = _stage(tmp_path, script, observer)
    assert result.returncode == 0 and result.status == 'passed'
    assert ack.is_file() and finished.is_file()
    assert len(received) == 1 and received[0].state == 'running'


def test_malformed_oversized_duplicate_and_split_frames_do_not_poison_next_event(tmp_path):
    valid = _event().to_dict()
    encoded = json.dumps(valid).encode('utf-8')
    duplicate = encoded[:-1] + b',"state":"completed"}\n'
    wrong_schema = {**valid, 'schema_version': True}
    with_private_blob = {**valid, 'provider_response': 'private-provider-response'}
    frames = [b'{invalid json}\n', b'\xff\n',
              b'x' * (transport.MAX_EVENT_BYTES + 17) + b'\n',
              duplicate, json.dumps(wrong_schema).encode() + b'\n',
              json.dumps(with_private_blob).encode() + b'\n',
              encoded[:31], encoded[31:] + b'\n']
    ack = tmp_path / 'valid-frame.ack'
    received = []
    def observer(event):
        received.append(event)
        ack.write_text('valid')
    script = '\n'.join([
        'import os',
        "fd = int(os.environ['NUMPILOT_PROGRESS_FD'])",
        'frames = ' + repr(frames),
        'for frame in frames: os.write(fd, frame)',
        _ack_wait_script(ack),
    ])
    result = _stage(tmp_path, script, observer)
    assert result.returncode == 0
    assert [item.to_dict() for item in received] == [valid]
    assert 'private-provider-response' not in json.dumps([item.to_dict() for item in received])


@pytest.mark.parametrize('fd_text', ['', '0', '1', '2', '-1', '3.0', '٣', '1000000000', 'invalid'])
def test_child_emitter_rejects_invalid_or_standard_fds_without_writing(monkeypatch, fd_text):
    monkeypatch.setenv(transport.PROGRESS_FD_ENV, fd_text)
    def forbidden(*args):
        pytest.fail('emitter attempted to write a forbidden descriptor')
    monkeypatch.setattr(transport.os, 'write', forbidden)
    assert transport.emit_child_event(_event()) is False


def test_child_emitter_never_writes_to_regular_file(tmp_path, monkeypatch):
    path = tmp_path / 'regular-output'
    path.write_bytes(b'original')
    with path.open('ab') as handle:
        monkeypatch.setenv(transport.PROGRESS_FD_ENV, str(handle.fileno()))
        assert transport.emit_child_event(_event()) is False
    assert path.read_bytes() == b'original'


def test_child_emitter_pipe_roundtrip_and_invalid_event_are_bounded(monkeypatch):
    reader, writer = os.pipe()
    try:
        monkeypatch.setenv(transport.PROGRESS_FD_ENV, str(writer))
        assert transport.emit_child_event(object()) is False
        value = _event()
        assert transport.emit_child_event(value) is True
        data = os.read(reader, transport.MAX_EVENT_BYTES + 1)
        assert len(data) <= transport.MAX_EVENT_BYTES and data.endswith(b'\n')
        assert RoleProgressEvent.from_dict(json.loads(data)).to_dict() == value.to_dict()
        monkeypatch.setenv(transport.PROGRESS_FD_ENV, str(reader))
        assert transport.emit_child_event(value) is False
    finally:
        os.close(reader)
        os.close(writer)


def test_full_pipe_drops_event_instead_of_blocking_model_worker(monkeypatch):
    reader, writer = os.pipe()
    try:
        os.set_blocking(writer, False)
        while True:
            try:
                os.write(writer, b'x' * transport.MAX_EVENT_BYTES)
            except BlockingIOError:
                break
        monkeypatch.setenv(transport.PROGRESS_FD_ENV, str(writer))
        assert transport.emit_child_event(_event()) is False
    finally:
        os.close(reader)
        os.close(writer)


def test_callback_exception_does_not_change_child_exit_status(tmp_path):
    received = []
    def broken_observer(event):
        received.append(event)
        raise RuntimeError('UI-only observer failure')
    script = '\n'.join([
        'from numpilot.progress import RoleProgressEvent',
        'from numpilot.progress_transport import emit_child_event',
        "assert emit_child_event(RoleProgressEvent('analyzer', 'analyzer', 'completed', '分析结束'))",
        'raise SystemExit(7)',
    ])
    result = _stage(tmp_path, script, broken_observer)
    assert result.returncode == 7 and result.status == 'failed'
    assert len(received) == 1
    assert 'observer' not in Path(result.stderr_path).read_text()


def _track_pipes(monkeypatch):
    records = []
    original = transport.ChildProgressPipe
    class TrackedPipe(original):
        def __init__(self, observer):
            super().__init__(observer)
            records.append((self, self.read_fd, self.write_fd))
    monkeypatch.setattr(transport, 'ChildProgressPipe', TrackedPipe)
    return records


def _assert_closed(records):
    assert len(records) == 1
    pipe, reader, writer = records[0]
    assert pipe.read_fd == pipe.write_fd == -1
    if pipe._thread is not None:
        assert not pipe._thread.is_alive()
    for descriptor in (reader, writer):
        with pytest.raises(OSError) as caught:
            os.fstat(descriptor)
        assert caught.value.errno == errno.EBADF


def test_timeout_terminates_child_and_closes_progress_fds(tmp_path, monkeypatch):
    records = _track_pipes(monkeypatch)
    result = _stage(tmp_path, 'import time; time.sleep(60)', lambda event: None, timeout=.2)
    assert result.status == 'timeout' and result.returncode != 0
    _assert_closed(records)


def test_launch_failure_closes_both_fds_without_raw_exception_content(tmp_path, monkeypatch):
    records = _track_pipes(monkeypatch)
    result = workflow._run_stage('launch', [str(tmp_path / 'private-missing-executable')], tmp_path,
        cwd=REPO, timeout_seconds=1, on_event=lambda event: None)
    assert result.status == 'launch_failed' and result.returncode is None
    _assert_closed(records)
    error = Path(result.stderr_path).read_text()
    assert 'FileNotFoundError' in error and 'private-missing-executable' not in error


def test_raw_stdout_and_stderr_are_logs_never_progress_events(tmp_path):
    valid_text = json.dumps(_event().to_dict())
    script = 'import sys\nprint(' + repr(valid_text) + ')\nprint("private-provider-secret", file=sys.stderr)'
    received = []
    result = _stage(tmp_path, script, received.append)
    assert result.status == 'passed' and received == []
    assert json.loads(Path(result.stdout_path).read_text()) == _event().to_dict()
    assert Path(result.stderr_path).read_text() == 'private-provider-secret\n'


def test_stage_without_observer_strips_inherited_progress_fd(tmp_path, monkeypatch):
    monkeypatch.setenv(transport.PROGRESS_FD_ENV, '12345')
    result = _stage(tmp_path, 'import os,json; print(json.dumps({"fd":os.getenv("NUMPILOT_PROGRESS_FD")}))', None)
    assert result.returncode == 0
    assert json.loads(Path(result.stdout_path).read_text()) == {'fd': None}


def test_workflow_observer_is_non_authoritative_and_keeps_strict_jsonl(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_text('import numpy as np\nresult=np.ones((2,2),dtype=np.float32)\n')
    def fake_stage(name, argv, root, *, cwd, timeout_seconds, on_event=None):
        emitter = ProgressEmitter(on_event)
        emitter.emit('analyzer', 'running', '读取调用契约')
        emitter.emit('analyzer', 'blocked', '输入资格不完整')
        stdout, stderr = root / (name + '.stdout.log'), root / (name + '.stderr.log')
        workflow._write_private(stdout, b'')
        workflow._write_private(stderr, b'private-provider-secret\n')
        return workflow.WorkflowStage(name=name, argv=tuple(argv), returncode=4, status='failed',
            elapsed_seconds=.01, stdout_path=str(stdout), stderr_path=str(stderr),
            stdout_sha256=workflow._sha256(b''), stderr_sha256=workflow._sha256(b'private-provider-secret\n'))
    monkeypatch.setattr(workflow, '_run_stage', fake_stage)
    baseline = workflow.run_ordinary_workflow(source, tmp_path / 'without-observer')
    received = []
    def observer(event):
        received.append(event)
        raise RuntimeError('observer exception must not affect result')
    observed = workflow.run_ordinary_workflow(source, tmp_path / 'with-observer', on_event=observer)
    assert baseline['status'] == observed['status'] == 'blocked'
    assert baseline['claims'] == observed['claims'] == {
        'model_generated': False, 'numerically_verified': False, 'strict_npu_performance_verified': False}
    assert baseline['stage_count'] == observed['stage_count'] == 1
    path = tmp_path / 'with-observer/workflow-progress.jsonl'
    serialized = [RoleProgressEvent.from_dict(json.loads(line)) for line in path.read_text().splitlines()]
    assert [event.to_dict() for event in serialized] == [event.to_dict() for event in received]
    assert [event.sequence for event in serialized] == list(range(1, len(serialized) + 1))
    assert 'private-provider-secret' not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / 'without-observer/workflow-progress.jsonl').exists()
    assert (tmp_path / 'with-observer/workflow-report.json').is_file()


def test_incomplete_final_frame_is_not_promoted_to_event(tmp_path):
    data = json.dumps(_event().to_dict()).encode()
    script = '\n'.join(['import os', "fd=int(os.environ['NUMPILOT_PROGRESS_FD'])", 'os.write(fd,' + repr(data) + ')'])
    received = []
    result = _stage(tmp_path, script, received.append)
    assert result.returncode == 0 and received == []
