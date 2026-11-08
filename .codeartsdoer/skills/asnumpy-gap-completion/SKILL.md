---
name: asnumpy-gap-completion
description: Complete a missing NumPy API through the NumPilot AsNumpy gap route. It validates the GapSpec, produces a source-compatible _core custom_ops overlay contract, binds a private custom OPP with explicit ACLNN symbol resolution, and stops fail-closed until numerical and performance gates pass.
---

# AsNumpy Gap Completion

Use this skill when a NumPy call is not directly available in the pinned AsNumpy
component and must become an `asnumpy.<name>` candidate. This skill owns the
integration boundary and qualification orchestration. CANNBot owns Ascend C or
`.asc` kernel generation and kernel repair. The skill never replaces the static
AsNumpy component in place.

## Input contract

Require all of the following before creating an overlay:

- a hash-bound `GapSpec` with the source API, public API, signature variants,
  dtype/shape/layout rules, NumPy oracle, target SoC and CANN version;
- a route-neutral operator artifact and its origin (`official_binary`,
  `official_source`, or `remote_llm`);
- a read-only AsNumpy source tree and component identity;
- for a custom OPP, the exact vendor root, `version.info`, ACLNN header,
  `libcust_opapi.so`, exported two-stage symbols, and provenance receipt;
- the selected CANNBot bundle and its license/skill hashes;
- an isolated run directory and an artifact-store reference.

Fail closed on missing files, identity drift, ambiguous manifests, unbound
vendor paths, or a mismatch between GapSpec and the candidate. Never infer an
operator name, dtype, shape, symbol, header, library, or fallback behavior.

## Required route

1. Resolve official candidates first. If an official installed ACLNN candidate
   satisfies the signed contract, use the official existing-module route and do
   not create a custom OPP.
2. For a generated custom operator, accept either the traditional
   `op_host/op_kernel` layout or one `.asc` implementation, but require exactly
   one `asnumpy.adapter.manifest.v1`.
3. The manifest must map `operator.<name>` to `asnumpy.<name>`, use the current
   `asnumpy._core` extension, and describe an incremental `custom_ops`
   object-library overlay. It must declare distinct ACLNN
   `GetWorkspaceSize`/execute symbols and non-empty conformance tests.
4. Custom ACLNN symbols must use
   `symbol_resolution.strategy=explicit-handle-dlsym`,
   `library_soname=libcust_opapi.so`, and `global_symbol_call=false`.
   Runtime code must resolve the logical symbols from the custom DSO with
   `dlopen(RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND)` and `dlsym`; `LD_PRELOAD` or
   link order is not an acceptable isolation mechanism.
5. Keep generated files in a run-scoped overlay. Never overwrite the pinned
   AsNumpy source, wheel, or component attachments during draft work.

## Gate order

Run these gates in order and persist a receipt for every gate:

1. GapSpec and candidate identity validation.
2. CANNBot bundle/manifest validation and custom OPP read-only provenance.
3. Overlay path, marker-edit, symbol-resolution, and source-context checks.
4. Isolated AsNumpy generator/CMake check and wheel build with custom operators
   required.
5. Fresh isolated installation and public `asnumpy.<name>` import.
6. NumPy oracle tests for ordinary values, semantic boundaries, all admitted
   shapes/dtypes, output shape/dtype, and rejected inputs.
7. Real NPU execution with no CPU fallback, including repeated execution and
   workspace/executor lifecycle checks.
8. Performance receipt under the declared single-core CPU/single-card NPU
   benchmark protocol, only after correctness passes.
9. Qualification and activation only after every required gate passes.

A build or import pass is not a numerical pass. A numerical pass is not a
performance pass. A generated candidate remains `generated_unverified` until
all gates are independently hash-bound.

## Artifact and failure contract

Store the immutable draft, overlay, custom OPP receipt, source/component
identity, skill hashes, build logs, runtime logs, NumPy results, NPU results,
performance receipt, and final qualification in the operator artifact store.
Each repair creates a new candidate bound to the previous report and raw
failure hashes; never mutate a prior draft.

Use these statuses:

- `generated_unverified`: model/CANNBot output accepted structurally;
- `built_unverified`: isolated build passed but runtime qualification is pending;
- `blocked`: a required gate failed or evidence is missing;
- `qualified`: all numerical and execution gates passed;
- `activated`: the qualified overlay was explicitly published for the run.

On any failure, preserve the raw diagnostic and stop at the failed gate.
`unknown`, `skipped`, CPU fallback, import-only execution, or a missing NPU
receipt never count as pass. Do not update the 150-example migration count
until the candidate is qualified.

## Interaction and model boundary

The skill does not call `llm.json` implicitly. A remote model request requires
explicit user authorization and a fresh run directory. The prompt includes
only hash-bound GapSpec, source context, CANNBot skill files, and declared
constraints; secrets, raw private configuration, and full response bodies stay
out of public receipts. The skill can validate a user/agent-authored draft
without relabeling it as a configured-model artifact.
