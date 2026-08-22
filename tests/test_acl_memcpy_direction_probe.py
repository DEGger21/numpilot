import json
import sqlite3

import pytest

from tools.acl_memcpy_direction_probe import (
    COPY_SPECS,
    PRODUCER,
    analyze_probe,
    classify_probe_rows,
    load_memcpy_rows,
)


def _row(byte_count, direction, timestamp):
    return {
        "timestamp": timestamp,
        "bytes": byte_count,
        "profiler_direction": direction,
    }


def test_copy_specs_have_unique_sizes_and_expected_acl_kinds():
    assert {item["acl_kind"] for item in COPY_SPECS} == {1, 2, 3}
    assert len({item["bytes"] for item in COPY_SPECS}) == len(COPY_SPECS)


def test_classification_accepts_a_discriminating_profiler_enum():
    rows = [
        _row(COPY_SPECS[0]["bytes"], 11, 100),
        _row(COPY_SPECS[1]["bytes"], 13, 200),
        _row(COPY_SPECS[2]["bytes"], 12, 300),
    ]

    result = classify_probe_rows(list(COPY_SPECS), rows)

    assert result["all_probe_calls_found_exactly_once"] is True
    assert result["direction_field_discriminates"] is True


def test_classification_reports_a_constant_profiler_field_without_guessing():
    rows = [
        _row(spec["bytes"], 0, index)
        for index, spec in enumerate(COPY_SPECS)
    ]

    result = classify_probe_rows(list(COPY_SPECS), rows)

    assert result["all_probe_calls_found_exactly_once"] is True
    assert result["direction_field_discriminates"] is False
    assert "not sufficient" in result["conclusion"]


def test_classification_fails_exact_match_when_probe_size_is_duplicated():
    rows = [
        _row(COPY_SPECS[0]["bytes"], 1, 100),
        _row(COPY_SPECS[0]["bytes"], 1, 101),
        _row(COPY_SPECS[1]["bytes"], 3, 200),
        _row(COPY_SPECS[2]["bytes"], 2, 300),
    ]

    result = classify_probe_rows(list(COPY_SPECS), rows)

    assert result["all_probe_calls_found_exactly_once"] is False
    assert result["direction_field_discriminates"] is False


def test_load_and_analyze_runtime_database(tmp_path):
    runtime_db = tmp_path / "runtime.db"
    with sqlite3.connect(runtime_db) as connection:
        connection.execute(
            "CREATE TABLE MemcpyInfo("
            "timestamp INTEGER, data_size INTEGER, memcpy_direction INTEGER)"
        )
        connection.executemany(
            "INSERT INTO MemcpyInfo VALUES (?, ?, ?)",
            [
                (10, COPY_SPECS[0]["bytes"], 0),
                (20, COPY_SPECS[1]["bytes"], 0),
                (30, COPY_SPECS[2]["bytes"], 0),
            ],
        )
    probe_receipt = tmp_path / "probe.json"
    probe_receipt.write_text(
        json.dumps({
            "producer": PRODUCER,
            "status": "pass",
            "copies": list(COPY_SPECS),
        }),
        encoding="utf-8",
    )

    rows = load_memcpy_rows(runtime_db)
    result = analyze_probe(probe_receipt=probe_receipt, runtime_db=runtime_db)

    assert len(rows) == 3
    assert result["status"] == "pass"
    assert result["direction_field_discriminates"] is False


def test_load_runtime_database_rejects_incomplete_schema(tmp_path):
    runtime_db = tmp_path / "runtime.db"
    with sqlite3.connect(runtime_db) as connection:
        connection.execute("CREATE TABLE MemcpyInfo(data_size INTEGER)")

    with pytest.raises(RuntimeError, match="missing columns"):
        load_memcpy_rows(runtime_db)
