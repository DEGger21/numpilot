"""Hash-bound observation of the inputs a gap API receives.

The observation stage executes the original NumPy program on the CPU and
records the concrete dtype/shape/contiguity of every tensor operand that
reached a gap callsite.  It never executes the device region, never records
tensor values and never grants support: it only replaces "unknown profiles"
in the taskbook with facts that were actually observed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Mapping, Tuple


OBSERVATION_SCHEMA = "numpilot.input-observation"
OBSERVATION_STATUSES = frozenset({
    "observed",
    "unsupported_dtype",
    "profile_missing",
    "nondeterministic",
    "observation_failed",
})
_FLAT_STATUSES = frozenset({"observed", "unsupported_dtype"})
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DTYPE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_QUALIFIED_API = re.compile(r"(?:numpy|operator)\.[A-Za-z][A-Za-z0-9_]{0,127}")
_TENSOR_FIELDS = frozenset({"dtype", "shape", "rank", "c_contiguous"})
_OPERAND_FIELDS = _TENSOR_FIELDS | {"name"}
_CALLSITE_FIELDS = frozenset({
    "qualified_api", "lineno", "col_offset", "positional_args",
    "keyword_args", "inputs", "unprofiled_operands", "output",
})


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _tensor_profile(value: Any, *, named: bool) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("observed tensor profile must be an object")
    expected = _OPERAND_FIELDS if named else _TENSOR_FIELDS
    if frozenset(value) != expected:
        raise ValueError("observed tensor profile fields are invalid")
    if not isinstance(value["dtype"], str) or not _DTYPE.fullmatch(value["dtype"]):
        raise ValueError("observed tensor dtype is invalid")
    shape = value["shape"]
    if (not isinstance(shape, list) or len(shape) > 64
            or any(not _is_int(size) or size < 0 or size > 2 ** 63 - 1 for size in shape)):
        raise ValueError("observed tensor shape is invalid")
    if value["rank"] != len(shape) or not _is_int(value["rank"]):
        raise ValueError("observed tensor rank does not match its shape")
    if type(value["c_contiguous"]) is not bool:
        raise ValueError("observed tensor contiguity is invalid")
    profile = {"dtype": value["dtype"], "shape": list(shape),
               "rank": value["rank"], "c_contiguous": value["c_contiguous"]}
    if named:
        if not isinstance(value["name"], str) or not value["name"]:
            raise ValueError("observed operand name is invalid")
        profile["name"] = value["name"]
    return profile


def _callsite(item: Any) -> Dict[str, Any]:
    if not isinstance(item, Mapping) or frozenset(item) != _CALLSITE_FIELDS:
        raise ValueError("observed callsite fields are invalid")
    if not isinstance(item["qualified_api"], str) or not _QUALIFIED_API.fullmatch(item["qualified_api"]):
        raise ValueError("observed callsite API is invalid")
    if not _is_int(item["lineno"]) or item["lineno"] < 1:
        raise ValueError("observed callsite line is invalid")
    if item["col_offset"] is not None and (not _is_int(item["col_offset"]) or item["col_offset"] < 0):
        raise ValueError("observed callsite column is invalid")
    if not _is_int(item["positional_args"]) or not 0 <= item["positional_args"] <= 128:
        raise ValueError("observed callsite arity is invalid")
    keywords = item["keyword_args"]
    if (not isinstance(keywords, list) or len(keywords) > 128
            or any(not isinstance(name, str) or not name for name in keywords)
            or keywords != sorted(set(keywords))):
        raise ValueError("observed callsite keywords are invalid")
    inputs = item["inputs"]
    if not isinstance(inputs, list) or len(inputs) > 128:
        raise ValueError("observed callsite inputs are invalid")
    operands = [_tensor_profile(value, named=True) for value in inputs]
    unprofiled = item["unprofiled_operands"]
    if (not isinstance(unprofiled, list) or len(unprofiled) > 128
            or any(not isinstance(name, str) or not name for name in unprofiled)):
        raise ValueError("observed callsite unprofiled operands are invalid")
    output = item["output"]
    return {
        "qualified_api": item["qualified_api"],
        "lineno": item["lineno"],
        "col_offset": item["col_offset"],
        "positional_args": item["positional_args"],
        "keyword_args": list(keywords),
        "inputs": operands,
        "unprofiled_operands": list(unprofiled),
        "output": None if output is None else _tensor_profile(output, named=False),
    }


@dataclass(frozen=True)
class InputObservation:
    """One receipt: what the gap callsites actually received on the CPU."""

    source_path: str
    source_sha256: str
    status: str
    callsites: Tuple[Dict[str, Any], ...]
    deterministic: bool
    notes: Tuple[str, ...] = ()
    executed: bool = True
    mode: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, str) or not self.source_path:
            raise ValueError("observation source path is invalid")
        if not isinstance(self.source_sha256, str) or not _HEX_SHA256.fullmatch(self.source_sha256):
            raise ValueError("observation source hash is invalid")
        if self.status not in OBSERVATION_STATUSES:
            raise ValueError("observation status is invalid")
        if self.mode not in {"contract", "script", "unknown"}:
            raise ValueError("observation mode is invalid")
        if type(self.deterministic) is not bool or type(self.executed) is not bool:
            raise ValueError("observation flags are invalid")
        if not isinstance(self.callsites, tuple) or len(self.callsites) > 64:
            raise ValueError("observation callsites are invalid")
        if not isinstance(self.notes, tuple) or len(self.notes) > 16:
            raise ValueError("observation notes are invalid")
        if any(not isinstance(note, str) or not note or len(note) > 512 for note in self.notes):
            raise ValueError("observation note text is invalid")
        seen = set()
        for item in self.callsites:
            normalized = _callsite(item)
            identity = (normalized["qualified_api"], normalized["lineno"],
                        normalized["positional_args"], tuple(normalized["keyword_args"]))
            if identity in seen:
                raise ValueError("observation callsites are duplicated")
            seen.add(identity)
            if normalized != item:
                raise ValueError("observation callsite is not normalized")
        if self.status in _FLAT_STATUSES:
            if not self.executed or not self.deterministic or not self.callsites:
                raise ValueError("flat observation status requires executed deterministic callsites")

    def to_dict(self) -> dict:
        return {
            "schema": OBSERVATION_SCHEMA,
            "schema_version": 1,
            "source": {"path": self.source_path, "sha256": self.source_sha256},
            "status": self.status,
            "mode": self.mode,
            "executed": self.executed,
            "deterministic": self.deterministic,
            "notes": list(self.notes),
            "callsites": [dict(item) for item in self.callsites],
        }


def _redacted(text: str) -> str:
    return re.sub(r"/[^\s'\"():]+", "<path>", text)[:512]


def _parse_worker_output(text: str, source_sha256: str, source_path: str) -> InputObservation:
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise ValueError("observation worker did not print one JSON object") from exc
    if not isinstance(payload, dict) or frozenset(payload) != {
        "schema", "schema_version", "source_sha256", "status", "mode",
        "executed", "deterministic", "notes", "callsites",
    }:
        raise ValueError("observation worker output fields are invalid")
    if (payload["schema"] != OBSERVATION_SCHEMA or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1):
        raise ValueError("observation worker schema is invalid")
    if payload["source_sha256"] != source_sha256:
        raise ValueError("observation worker ran a different source revision")
    if payload["status"] not in OBSERVATION_STATUSES:
        raise ValueError("observation worker status is invalid")
    if type(payload["executed"]) is not bool or type(payload["deterministic"]) is not bool:
        raise ValueError("observation worker flags are invalid")
    notes = payload["notes"]
    if (not isinstance(notes, list) or len(notes) > 16
            or any(not isinstance(note, str) or not note or len(note) > 512 for note in notes)):
        raise ValueError("observation worker notes are invalid")
    raw = payload["callsites"]
    if not isinstance(raw, list) or len(raw) > 64:
        raise ValueError("observation worker callsites are invalid")
    return InputObservation(
        source_path=source_path,
        source_sha256=source_sha256,
        status=payload["status"],
        callsites=tuple(dict(item) for item in raw),
        deterministic=payload["deterministic"],
        notes=tuple(notes),
        executed=payload["executed"],
        mode=payload["mode"],
    )


def observe_gap_inputs(source_path: str, *, gap_apis, timeout: float = 120.0,
                       source_label: str | None = None) -> InputObservation:
    """Run the source on the CPU and record tensor operands at gap callsites.

    The worker runs twice and must observe the same dtype/shape/contiguity at
    every callsite.  Any failure is returned as a receipt with an explicit
    status; it is never silently converted into "no profiles needed".
    """
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 1.0 <= timeout <= 3600.0:
        raise ValueError("observation timeout must be between 1 and 3600 seconds")
    apis = sorted({str(api) for api in gap_apis})
    if not apis or len(apis) > 64 or any(not _QUALIFIED_API.fullmatch(api) for api in apis):
        raise ValueError("observation requires bounded qualified gap APIs")
    path = Path(source_path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError("observation source cannot be read") from exc
    source_sha256 = hashlib.sha256(data).hexdigest()
    label = source_label if source_label is not None else str(path)
    repository_root = str(Path(__file__).resolve().parents[2])
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = repository_root + (os.pathsep + existing if existing else "")
    command = [sys.executable, "-m", "numpilot.gaps._input_observation_worker",
               str(path.resolve()), json.dumps(apis, ensure_ascii=True)]
    try:
        completed = subprocess.run(
            command, cwd=str(path.resolve().parent), env=environment,
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return InputObservation(
            source_path=label, source_sha256=source_sha256, status="observation_failed",
            callsites=(), deterministic=False, executed=False,
            notes=("observation timed out after %ss" % timeout,),
        )
    except OSError:
        return InputObservation(
            source_path=label, source_sha256=source_sha256, status="observation_failed",
            callsites=(), deterministic=False, executed=False,
            notes=("observation worker could not be started",),
        )
    if completed.returncode != 0:
        return InputObservation(
            source_path=label, source_sha256=source_sha256, status="observation_failed",
            callsites=(), deterministic=False, executed=False,
            notes=("observation worker exited with status %d" % completed.returncode,
                   _redacted((completed.stderr or "").strip().splitlines()[-1]
                             if (completed.stderr or "").strip() else "no diagnostic")),
        )
    observation = _parse_worker_output(completed.stdout, source_sha256, label)
    return observation
