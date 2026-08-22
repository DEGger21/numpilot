#!/usr/bin/env python3
"""Issue distinguishable ACL memcpy calls and audit a CANN profiler database.

The CANN compact ``MemcpyInfo`` record stores a numeric direction field.  This
tool deliberately uses a unique byte count for each requested ACL direction so
the profiler output can be matched without assuming that its compact enum is
identical to ``aclrtMemcpyKind``.

Typical use::

    msprof --output=/path/to/profiler-root --application=python \
      --application-args="tools/acl_memcpy_direction_probe.py run --out /path/to/run"

    python tools/acl_memcpy_direction_probe.py analyze \
      --probe-receipt /path/to/run/evidence.json \
      --runtime-db /path/to/PROF_.../host/sqlite/runtime.db \
      --out /path/to/analysis
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sqlite3
import traceback


SCHEMA_VERSION = "1.0"
PRODUCER = "numpilot-acl-memcpy-direction-probe-v1"
ALLOCATION_BYTES = 4096
COPY_SPECS = (
    {"direction": "host_to_device", "acl_kind": 1, "bytes": 4093},
    {"direction": "device_to_device", "acl_kind": 3, "bytes": 2053},
    {"direction": "device_to_host", "acl_kind": 2, "bytes": 1021},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_acl(library):
    library.aclInit.argtypes = [ctypes.c_char_p]
    library.aclInit.restype = ctypes.c_int
    library.aclFinalize.argtypes = []
    library.aclFinalize.restype = ctypes.c_int
    library.aclrtSetDevice.argtypes = [ctypes.c_int]
    library.aclrtSetDevice.restype = ctypes.c_int
    library.aclrtResetDevice.argtypes = [ctypes.c_int]
    library.aclrtResetDevice.restype = ctypes.c_int
    library.aclrtMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    library.aclrtMalloc.restype = ctypes.c_int
    library.aclrtFree.argtypes = [ctypes.c_void_p]
    library.aclrtFree.restype = ctypes.c_int
    library.aclrtMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    library.aclrtMemcpy.restype = ctypes.c_int
    library.aclrtSynchronizeDevice.argtypes = []
    library.aclrtSynchronizeDevice.restype = ctypes.c_int
    return library


def _check(code: int, call: str) -> None:
    if code != 0:
        raise RuntimeError(f"{call} failed with ACL error {code}")


def issue_probe(*, device: int, library_name: str) -> dict:
    """Run three verified copies whose byte counts identify their directions."""

    library = _configure_acl(ctypes.CDLL(library_name, mode=ctypes.RTLD_GLOBAL))
    device_a = ctypes.c_void_p()
    device_b = ctypes.c_void_p()
    initialized = False
    device_set = False
    cleanup_errors = []
    try:
        _check(library.aclInit(None), "aclInit")
        initialized = True
        _check(library.aclrtSetDevice(device), "aclrtSetDevice")
        device_set = True
        # ACL_MEM_MALLOC_HUGE_FIRST == 0 in acl_rt.h.
        _check(
            library.aclrtMalloc(ctypes.byref(device_a), ALLOCATION_BYTES, 0),
            "aclrtMalloc(device_a)",
        )
        _check(
            library.aclrtMalloc(ctypes.byref(device_b), ALLOCATION_BYTES, 0),
            "aclrtMalloc(device_b)",
        )

        source_bytes = bytes((index * 17 + 29) % 256 for index in range(ALLOCATION_BYTES))
        source = ctypes.create_string_buffer(source_bytes, ALLOCATION_BYTES)
        received = ctypes.create_string_buffer(ALLOCATION_BYTES)

        h2d, d2d, d2h = COPY_SPECS
        _check(
            library.aclrtMemcpy(
                device_a,
                ALLOCATION_BYTES,
                ctypes.cast(source, ctypes.c_void_p),
                h2d["bytes"],
                h2d["acl_kind"],
            ),
            "aclrtMemcpy(host_to_device)",
        )
        _check(
            library.aclrtMemcpy(
                device_b,
                ALLOCATION_BYTES,
                device_a,
                d2d["bytes"],
                d2d["acl_kind"],
            ),
            "aclrtMemcpy(device_to_device)",
        )
        _check(
            library.aclrtMemcpy(
                ctypes.cast(received, ctypes.c_void_p),
                ALLOCATION_BYTES,
                device_b,
                d2h["bytes"],
                d2h["acl_kind"],
            ),
            "aclrtMemcpy(device_to_host)",
        )
        _check(library.aclrtSynchronizeDevice(), "aclrtSynchronizeDevice")

        copied = received.raw[: d2h["bytes"]]
        expected = source_bytes[: d2h["bytes"]]
        if copied != expected:
            raise RuntimeError("round-trip payload differs after H2D -> D2D -> D2H")

        return {
            "schema_version": SCHEMA_VERSION,
            "producer": PRODUCER,
            "status": "pass",
            "device": device,
            "acl_library": library_name,
            "allocation_bytes": ALLOCATION_BYTES,
            "copies": [dict(item) for item in COPY_SPECS],
            "payload_verified": True,
            "script_sha256": _sha256(Path(__file__).resolve()),
        }
    finally:
        for pointer, label in ((device_b, "device_b"), (device_a, "device_a")):
            if pointer.value:
                code = library.aclrtFree(pointer)
                if code != 0:
                    cleanup_errors.append(f"aclrtFree({label})={code}")
        if device_set:
            code = library.aclrtResetDevice(device)
            if code != 0:
                cleanup_errors.append(f"aclrtResetDevice={code}")
        if initialized:
            code = library.aclFinalize()
            if code != 0:
                cleanup_errors.append(f"aclFinalize={code}")
        if cleanup_errors:
            raise RuntimeError("ACL cleanup failed: " + ", ".join(cleanup_errors))


def load_memcpy_rows(runtime_db: Path) -> list[dict]:
    """Load only the compact fields needed for a direction audit."""

    if not runtime_db.is_file():
        raise FileNotFoundError(f"runtime database is missing: {runtime_db}")
    with sqlite3.connect(f"file:{runtime_db}?mode=ro", uri=True) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(MemcpyInfo)")
        }
        required = {"timestamp", "data_size", "memcpy_direction"}
        if not required.issubset(columns):
            raise RuntimeError(
                "MemcpyInfo schema is missing columns: "
                + ", ".join(sorted(required - columns))
            )
        rows = connection.execute(
            "SELECT timestamp, data_size, memcpy_direction "
            "FROM MemcpyInfo ORDER BY timestamp"
        ).fetchall()
    return [
        {
            "timestamp": int(timestamp),
            "bytes": int(data_size),
            "profiler_direction": int(direction),
        }
        for timestamp, data_size, direction in rows
    ]


def classify_probe_rows(copies: list[dict], rows: list[dict]) -> dict:
    """Match probe calls by unique byte count and test enum discriminability."""

    sizes = [int(copy["bytes"]) for copy in copies]
    if len(sizes) != len(set(sizes)):
        raise ValueError("probe copy byte counts must be unique")
    matches = []
    for copy in copies:
        matching = [row for row in rows if row["bytes"] == int(copy["bytes"])]
        matches.append({
            **copy,
            "profiler_matches": len(matching),
            "observed_profiler_directions": sorted({
                row["profiler_direction"] for row in matching
            }),
            "timestamps": [row["timestamp"] for row in matching],
        })

    exactly_once = all(match["profiler_matches"] == 1 for match in matches)
    singleton_codes = [
        match["observed_profiler_directions"][0]
        for match in matches
        if len(match["observed_profiler_directions"]) == 1
    ]
    direction_field_discriminates = (
        exactly_once
        and len(singleton_codes) == len(matches)
        and len(set(singleton_codes)) == len(matches)
    )
    return {
        "matches": matches,
        "all_probe_calls_found_exactly_once": exactly_once,
        "direction_field_discriminates": direction_field_discriminates,
        "conclusion": (
            "compact profiler direction codes distinguish H2D/D2D/D2H"
            if direction_field_discriminates
            else "compact profiler direction field is not sufficient to distinguish H2D/D2D/D2H"
        ),
    }


def analyze_probe(*, probe_receipt: Path, runtime_db: Path) -> dict:
    probe = json.loads(probe_receipt.read_text(encoding="utf-8"))
    if probe.get("producer") != PRODUCER or probe.get("status") != "pass":
        raise RuntimeError("probe receipt is not a passing trusted memcpy probe")
    rows = load_memcpy_rows(runtime_db)
    classification = classify_probe_rows(probe["copies"], rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": "numpilot-acl-memcpy-profiler-audit-v1",
        "status": (
            "pass"
            if classification["all_probe_calls_found_exactly_once"]
            else "fail"
        ),
        "probe_receipt": str(probe_receipt.resolve()),
        "probe_receipt_sha256": _sha256(probe_receipt),
        "runtime_db": str(runtime_db.resolve()),
        "runtime_db_sha256": _sha256(runtime_db),
        "total_memcpy_rows": len(rows),
        **classification,
    }


def _render_markdown(receipt: dict) -> str:
    lines = [
        "# ACL memcpy profiler probe",
        "",
        f"Status: `{receipt['status']}`",
        "",
    ]
    if "matches" not in receipt:
        lines.extend([
            f"- Device: `{receipt.get('device')}`",
            f"- Payload verified: `{str(receipt.get('payload_verified', False)).lower()}`",
            f"- Copies: `{len(receipt.get('copies', []))}`",
        ])
        return "\n".join(lines) + "\n"
    lines.extend([
        f"- Memcpy rows: `{receipt['total_memcpy_rows']}`",
        "- Direction field discriminates: "
        f"`{str(receipt['direction_field_discriminates']).lower()}`",
        f"- Conclusion: {receipt['conclusion']}",
        "",
        "| Requested direction | ACL kind | Bytes | Matches | Profiler code |",
        "|---|---:|---:|---:|---|",
    ])
    for match in receipt["matches"]:
        lines.append(
            f"| `{match['direction']}` | {match['acl_kind']} | {match['bytes']} | "
            f"{match['profiler_matches']} | `{match['observed_profiler_directions']}` |"
        )
    return "\n".join(lines) + "\n"


def _write_receipt(output: Path, receipt: dict) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(_render_markdown(receipt), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="issue known ACL memcpy calls")
    run_parser.add_argument("--device", type=int, default=0)
    run_parser.add_argument("--library", default="libascendcl.so")
    run_parser.add_argument("--out", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze", help="match a probe receipt to profiler runtime.db"
    )
    analyze_parser.add_argument("--probe-receipt", required=True)
    analyze_parser.add_argument("--runtime-db", required=True)
    analyze_parser.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    output = Path(args.out).resolve()
    if output.exists():
        parser.error(f"--out must not already exist: {output}")
    try:
        if args.command == "run":
            if args.device < 0:
                parser.error("--device must be non-negative")
            receipt = issue_probe(device=args.device, library_name=args.library)
        else:
            receipt = analyze_probe(
                probe_receipt=Path(args.probe_receipt).resolve(),
                runtime_db=Path(args.runtime_db).resolve(),
            )
    except Exception as error:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "producer": PRODUCER,
            "status": "fail",
            "failure_phase": args.command,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
        _write_receipt(output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 1

    _write_receipt(output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
