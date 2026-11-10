"""A native call counter must never masquerade as profiler device evidence."""
import csv
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'cat_profile', Path(__file__).parents[1] / 'tools' / 'concatenate_kernel_profile.py')
profile = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profile)


def trace(tmp_path, rows, fields=None):
    path = tmp_path / 'op_summary.csv'
    fields = fields or ['OP Type', 'Op Name', 'Task Type', 'Task Duration(us)']
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    return path


def test_counts_only_measured_cat_device_tasks(tmp_path):
    path = trace(tmp_path, [
        ['ConcatD', 'aclnnCat_ConcatD_ConcatD', 'AI_VECTOR_CORE', '1.2'],
        ['ConcatD', 'aclnnCat_ConcatD_ConcatD', 'AI_CORE', '2.5'],
        ['Scatter', 'aclnnScatter', 'AI_VECTOR_CORE', '1'],
        ['ConcatD', 'aclnnCat_ConcatD_ConcatD', 'HOST', '1'],
        ['ConcatD', 'unrelated_concat', 'AI_VECTOR_CORE', '1'],
    ])
    assert profile.summarize_trace(path) == (5, {'ConcatD': 4, 'Scatter': 1}, 2)


@pytest.mark.parametrize('duration', ['0', '-1', 'nan', 'inf', 'unknown'])
def test_rejects_missing_or_invalid_measurement(tmp_path, duration):
    path = trace(tmp_path, [['ConcatD', 'aclnnCat_ConcatD_ConcatD', 'AI_VECTOR_CORE', duration]])
    with pytest.raises(ValueError):
        profile.summarize_trace(path)


def test_rejects_api_call_csv_as_device_evidence(tmp_path):
    path = trace(tmp_path, [['aclnnCat', '4']], ['Name', 'Count'])
    with pytest.raises(ValueError, match='device task columns'):
        profile.summarize_trace(path)


def test_empty_trace_cannot_claim_device_execution(tmp_path):
    assert profile.summarize_trace(trace(tmp_path, [])) == (0, {}, 0)
