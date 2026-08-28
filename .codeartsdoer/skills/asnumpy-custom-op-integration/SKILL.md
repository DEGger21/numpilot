---
name: asnumpy-custom-op-integration
description: Integrate either a verified installed official ACLNN candidate or a built custom operator into AsNumpy using the detected engineering architecture, isolated overlays, wheel build, and NPU tests. Do not use this skill to write or optimize an Ascend C kernel.
---

# AsNumpy Operator Integration

Use this skill only for the boundary between a verified operator candidate and
AsNumpy. The candidate route is authoritative:

- `official_binary` reuses an installed ACLNN kernel and must follow
  [references/official-aclnn-existing-module.md](references/official-aclnn-existing-module.md);
- `official_source` uses a separately materialized, built, and qualified
  pinned ops-nn snapshot;
- `remote_llm` uses a separately built custom OPP produced by the external
  CANNBot workflow.

Never regenerate an official kernel. Ascend C kernel generation, compilation,
and kernel debugging for `remote_llm` belong to CANNBot. Do not copy CANNBot
content into this skill.

## Required inputs

- the immutable route-neutral operator candidate and adapter contract;
- for `official_binary`, the exact installed CANN header, shared library,
  symbols, catalog mapping, and semantic-review receipts;
- for a custom OPP, its exact private vendor directory, ACLNN headers,
  exported symbols, and `libcust_opapi.so`;
- a read-only AsNumpy source tree;
- the NumPy oracle and admitted dtype/shape rules;
- raw build, runtime, or precision diagnostics for a repair iteration.

Fail closed if an input is absent or its identity/hash does not match its
report. Never guess a vendor, header, symbol, dtype, shape, or output contract.

## Route by target AsNumpy architecture

If the contract strategy is `official_aclnn_existing_module`, read
[references/official-aclnn-existing-module.md](references/official-aclnn-existing-module.md)
and obey the supplied reviewed existing-module context plan. Do not select a
custom-op manifest/manual route, add `cust_opapi`, or invent a private vendor.

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

1. Inspect the exact route, target context plan, and one enabled implementation
   in the same AsNumpy module.
2. Inventory the selected official binary or private OPP without modifying it.
   Bind every header, library, symbol, and semantic restriction to receipts.
3. Have the configured user model produce an isolated overlay. A local
   orchestrator may validate and mechanically apply it but must not author or
   repair model-owned integration files.
4. For a manifest route, regenerate glue and run the generator's check mode.
   For an existing-module route, reject every file outside its reviewed overlay
   allow-list.
5. Build and install the wheel in an isolated environment. A custom-OPP route
   must keep custom operators required; an official-binary route must link only
   the verified installed CANN libraries and must not add a custom OPP.
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
