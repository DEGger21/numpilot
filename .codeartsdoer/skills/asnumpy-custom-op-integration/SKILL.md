---
name: asnumpy-custom-op-integration
description: Integrate a built and precision-targeted Ascend custom operator into AsNumpy using its manifest generator, private OPP loader, wheel build, and NPU tests. Use after operator generation; do not use this skill to write or optimize the Ascend C operator itself.
---

# AsNumpy Custom-Operator Integration

Use this skill only for the boundary between a separately built custom OPP and
AsNumpy. Ascend C semantics, kernel implementation, compilation, and kernel
debugging belong to the external CANNBot workflow. Do not copy CANNBot content
into this skill.

## Required inputs

- the immutable operator artifact and its model-generated adapter contract;
- the installed private OPP root, including its exact vendor directory, ACLNN
  headers, exported symbols, and `libcust_opapi.so`;
- a read-only AsNumpy source tree;
- the NumPy oracle and admitted dtype/shape rules;
- raw build, runtime, or precision diagnostics for a repair iteration.

Fail closed if an input is absent or its identity/hash does not match its
report. Never guess a vendor, header, symbol, dtype, shape, or output contract.

## Route by target AsNumpy architecture

If `scripts/gen_custom_ops.py` and `csrc/custom_ops/registry/` exist, read
[references/manifest-workflow.md](references/manifest-workflow.md) and use the
manifest path. This is the preferred path: generate only the operator manifest
and its NPU precision test, then run the repository generator mechanically.
Never hand-edit generator-owned C++, pybind, Python export, or CMake outputs.

Only when the target has no manifest generator, read
[references/legacy-case-lessons.md](references/legacy-case-lessons.md). Inspect
the target tree before adapting the older manual call chain; do not assume its
file names or build layout are unchanged.

For build, runtime, precision, activation, or repair decisions, read
[references/qualification-and-debug.md](references/qualification-and-debug.md).

## Workflow invariants

1. Inspect the target generator/schema and one enabled reference operator.
2. Inventory the private OPP without modifying it. Bind `opp_vendor` to the
   actual vendor directory and bind ACLNN names to installed headers and
   exported symbols.
3. Have the configured user model produce an isolated overlay. A local
   orchestrator may validate and mechanically apply it but must not author or
   repair model-owned integration files.
4. Regenerate glue and run the generator's check mode. Generated files must be
   reproducible from manifests.
5. Build a wheel with custom operators required, not auto-disabled. Install it
   into an isolated environment and keep every custom OPP private to the run.
6. Run real NPU tests against the NumPy oracle. A skip, CPU fallback, import-only
   check, or unverified execution source is not a precision pass.
7. Return raw compiler/runtime/precision logs to the configured model for each
   repair. Publish a new hash-bound overlay; never mutate the prior overlay.
8. Activate `asnumpy.<name>` only after build, public API, NPU execution, and
   numerical gates all pass.

## Correctness versus performance

Ask before generation whether the user wants `accuracy` or `performance`.
Recommend `accuracy`: stop after semantic, build, runtime, and numerical
correctness. If the user requests `performance`, warn that measured optimization
can require many long iterations and ask again after correctness passes before
starting it. Never trade away correctness for performance.
