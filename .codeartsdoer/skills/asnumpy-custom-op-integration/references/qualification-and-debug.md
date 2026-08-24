# Qualification and diagnostic loop

Use this reference after an integration overlay exists or when a prior build,
runtime, or precision attempt failed.

## Evidence sequence

1. Preserve the model overlay and its report as immutable input.
2. Mechanically apply it to an isolated AsNumpy worktree.
3. Regenerate glue and run the generator's check mode.
4. Inventory every required private vendor, header, library, symbol, and kernel
   metadata path.
5. Build with custom operators required (`ON` or the target's strict
   equivalent). AUTO/off builds cannot qualify the operator.
6. Build a wheel with bounded parallelism and install it into a fresh isolated
   Python environment.
7. With the same private OPP roots, import the public API and run the dedicated
   real-NPU tests.
8. Compare outputs, shapes, and dtypes with the NumPy oracle and save the raw
   results plus execution provenance.
9. Activate only after every required gate passes.

Use one CPU thread for CPU comparison programs and record the observed thread
pool. This is separate from build parallelism and from NPU performance claims.

## Failure routing

Return the unedited diagnostic and relevant current files to the configured
user model. The orchestrator can select and validate files but must not repair
model-owned sources.

- Generator/schema failure: repair the manifest or declared manual fragments.
- Missing vendor/header/library: provide the raw build log plus a read-only OPP
  inventory. Repair the manifest or operator package; do not guess paths.
- Missing dynamic symbol/dependency: provide `nm`/`ldd` output and loader error.
- Import or public-export failure: provide wheel build/install/import logs and
  generated registry evidence.
- ACLNN/runtime/kernel failure: use the external CANNBot runtime-debug workflow
  for operator code; use this skill only if the integration ABI/path is wrong.
- Numerical mismatch: use the external CANNBot precision workflow for kernel
  code. Repair an AsNumpy test only when the test contradicts the signed NumPy
  oracle; never loosen tolerances merely to pass.

Each repair must create a new candidate bound to the previous report hash, raw
diagnostic hash, selected skill-file hashes, response hash, and remote receipt.

## Pass criteria

A customer-ready accuracy-first integration requires all of the following:

- deterministic generator check passes;
- strict custom-op wheel build passes;
- isolated wheel installation and public import pass;
- the requested test does not skip and invokes the custom NPU path;
- NumPy comparison passes for admitted dtypes/shapes and edge cases;
- no CPU fallback or unknown execution source;
- artifact, overlay, private OPP, wheel, logs, and reports are hash-bound.

Performance optimization is outside the default pass criteria. If requested,
start it only after the above gates pass and after warning the user that the
measured optimization loop may be lengthy.
