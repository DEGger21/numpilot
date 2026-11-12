"""Isolated CPU worker: observe the tensors a gap API actually receives.

Protocol: ``python -m numpilot.gaps._input_observation_worker SOURCE APIS``
where ``APIS`` is a JSON list of qualified NumPy names.  The worker executes
the program twice with recorders installed, compares dtype/shape/contiguity
fingerprints and prints exactly one JSON object on stdout.  Program output is
redirected to stderr so it cannot corrupt the protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path


_SCALARS = (int, float, bool, complex, str, bytes, bytearray, type(None))


def _note(message):
    return re.sub(r"/[^\s'\"():]+", "<path>", str(message))[:500]


def _profile(value):
    from numpilot.capabilities import tensor_profile_from_value

    profile = tensor_profile_from_value(value)
    if profile is None:
        return None
    return {"dtype": profile.dtype, "shape": list(profile.shape),
            "rank": profile.rank, "c_contiguous": profile.c_contiguous}


def _is_scalar(value):
    if isinstance(value, _SCALARS):
        return True
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        return False
    return isinstance(value, np.generic)


def _collect(args, kwargs):
    operands = []
    unprofiled = []
    for index, value in enumerate(args):
        name = "arg[%d]" % index
        if isinstance(value, (list, tuple)):
            for sub, item in enumerate(value):
                profile = _profile(item)
                if profile is None:
                    if not _is_scalar(item):
                        unprofiled.append("%s[%d]" % (name, sub))
                    continue
                profile["name"] = "%s[%d]" % (name, sub)
                operands.append(profile)
            continue
        profile = _profile(value)
        if profile is None:
            if not _is_scalar(value):
                unprofiled.append(name)
            continue
        profile["name"] = name
        operands.append(profile)
    for key in sorted(kwargs):
        value = kwargs[key]
        if isinstance(value, (list, tuple)):
            for sub, item in enumerate(value):
                profile = _profile(item)
                if profile is None:
                    if not _is_scalar(item):
                        unprofiled.append("%s[%d]" % (key, sub))
                    continue
                profile["name"] = "%s[%d]" % (key, sub)
                operands.append(profile)
            continue
        profile = _profile(value)
        if profile is None:
            if not _is_scalar(value):
                unprofiled.append(key)
            continue
        profile["name"] = key
        operands.append(profile)
    return operands, unprofiled


def _execute(source_file, source_text, api_names, numpy):
    records = []
    originals = {}
    source_real = os.path.realpath(source_file)

    def make_recorder(api_name, target):
        def recorder(*args, **kwargs):
            caller = sys._getframe(1)
            try:
                same_file = os.path.realpath(caller.f_code.co_filename) == source_real
            except (OSError, ValueError):
                same_file = False
            record = None
            if same_file:
                operands, unprofiled = _collect(args, kwargs)
                record = {
                    "qualified_api": api_name,
                    "lineno": caller.f_lineno,
                    "col_offset": None,
                    "positional_args": len(args),
                    "keyword_args": sorted(kwargs),
                    "inputs": operands,
                    "unprofiled_operands": unprofiled,
                    "output": None,
                }
                records.append(record)
            result = target(*args, **kwargs)
            if record is not None:
                record["output"] = _profile(result)
            return result

        return recorder

    for name in api_names:
        short = name.split(".", 1)[1]
        original = getattr(numpy, short, None)
        if not callable(original):
            continue
        originals[short] = original
        setattr(numpy, short, make_recorder(name, original))
    namespace = {"__name__": "numpilot_observed_program", "__file__": source_real}
    mode = "script"
    try:
        exec(compile(source_text, source_real, "exec"), namespace)
        build_inputs = namespace.get("build_inputs")
        forward = namespace.get("forward")
        if callable(build_inputs) and callable(forward):
            mode = "contract"
            inputs = build_inputs()
            if not isinstance(inputs, dict):
                raise TypeError("build_inputs must return a mapping")
            forward(**inputs)
    finally:
        for short, original in originals.items():
            setattr(numpy, short, original)
    return records, mode


def _fingerprint(records):
    return json.dumps(sorted(records, key=lambda item: (
        item["lineno"], item["qualified_api"], item["positional_args"],
        tuple(item["keyword_args"]))), sort_keys=True, ensure_ascii=True)


def _has_foreign_dtype(records):
    return any(operand["dtype"] != "float32"
               for record in records for operand in record["inputs"])


def _emit(payload):
    sys.stdout.write(json.dumps(payload, sort_keys=True, ensure_ascii=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        sys.stderr.write("usage: worker SOURCE APIS_JSON\n")
        return 2
    source_file = os.path.realpath(argv[0])
    try:
        api_names = json.loads(argv[1])
        if (not isinstance(api_names, list) or not api_names
                or any(not isinstance(name, str) for name in api_names)):
            raise ValueError
    except ValueError:
        sys.stderr.write("worker: APIS must be a non-empty JSON list of names\n")
        return 2
    try:
        text = Path(source_file).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        _emit({"schema": "numpilot.input-observation", "schema_version": 1,
               "source_sha256": "0" * 64, "status": "observation_failed",
               "mode": "unknown", "executed": False, "deterministic": False,
               "notes": ["observation source cannot be read"], "callsites": []})
        return 0
    source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    import numpy

    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    status, deterministic, executed = "observation_failed", False, False
    notes = []
    first = []
    try:
        first, mode = _execute(source_file, text, api_names, numpy)
        executed = True
        second, second_mode = _execute(source_file, text, api_names, numpy)
    except (Exception, SystemExit) as exc:  # one bad program must not abort the sweep
        notes.append(_note("program execution failed: %s: %s"
                           % (type(exc).__name__, str(exc))))
        first = []
    finally:
        sys.stdout = real_stdout
    if notes:
        status = "observation_failed"
    else:
        if _fingerprint(first) != _fingerprint(second):
            status, deterministic = "nondeterministic", False
            notes.append("two executions observed different tensor profiles")
        elif not first:
            status, deterministic = "profile_missing", True
            notes.append("no gap API call was reached in the observed file")
        elif _has_foreign_dtype(first):
            status, deterministic = "unsupported_dtype", True
            notes.append("observed operand dtype is outside the float32 development scope")
        else:
            status, deterministic = "observed", True
            notes.append("two CPU executions agreed on every observed tensor profile")
    if status == "observation_failed":
        mode = "unknown"
    _emit({"schema": "numpilot.input-observation", "schema_version": 1,
           "source_sha256": source_sha256, "status": status,
           "mode": mode, "executed": executed, "deterministic": deterministic,
           "notes": notes, "callsites": first if executed else []})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
