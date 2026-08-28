---
name: numpilot-migration-gate
description: Run NumPilot's five-state, configured-model ordinary NumPy-to-AsNumpy/NPU migration and evidence workflow. Use for analysis, migration, explicit numeric verification, gap routing, or performance-evidence review on Ascend NPU.
---

# NumPilot Migration Gate

## Trust boundary

The default and competition-facing input is an ordinary, self-contained Python file with standard
`import numpy as np`. Do not require `build_inputs()` or `forward()`. Static admission, configured-model
generation, explicit execution, NPU provenance, and performance are separate claims with separate receipts.

The competition path must use the competition environment's real configured GLM through the supported
OpenAI-compatible protocol; the exact model name remains bound by configuration and receipt. Local AST logic may construct and
check the exact expected namespace rewrite, but it must never generate the submitted artifact or replace a
failed model call. Do not use `--engine ast`, `--mode auto`, or `--mode contract` for a competition ordinary
artifact.

## Inputs

- ordinary source Python file;
- a new, dedicated run directory outside the source tree;
- one or more explicit numeric/bool global output names if execution is authorized;
- target backend/environment, normally `npu`, plus exact SoC/CANN/runtime identity;
- fallback policy and optional approved GapSpec when the five-state analysis requires them;
- optional performance request, whose timing boundary must be stated before measurement.

Never put a model token, `llm.json`, response body, private OPP, or unredacted environment dump in the
repository or evidence bundle.

## Ordinary workflow

### 1. Read and inspect without execution

Read `README.md`, `docs/numpilot-technical-review.md`, `docs/codearts-integration.md`, and the relevant
capability/provenance records. Ask `numpilot-analyzer` for a read-only review, then have the deterministic
execution plane publish a write-once receipt:

```bash
python -m numpilot analyze SOURCE --json --out RUN_DIR/qualified-analysis.json
```

This command must record the current source path/SHA-256, callsites, runtime identity and exactly one of:

- `DIRECT`: all calls are qualified for the bound strict/scoped NPU runtime;
- `FALLBACK_CONSENT`: an audited CPU/host fallback requires explicit user choice;
- `GAP_REQUIRED`: an API is genuinely absent, or the user rejected an audited fallback;
- `CAPABILITY_AUDIT_REQUIRED`: a callable has unknown/drifted/unbound provenance;
- `BLOCKED`: the Python/NumPy surface cannot currently be migrated safely.

Analysis is inert: `actual_remote_call=false` and `execution_performed=false`. `--require-direct` may be
added in CI, but a nonzero result must not erase the completed non-DIRECT receipt.

### 2. Dispatch the five states without collapsing them

- On `DIRECT`, continue to configured-model migration.
- On `FALLBACK_CONSENT`, pause for explicit acceptance or rejection. Acceptance may produce a compatibility
  artifact only; it is always `strict_npu=false` and `performance_valid=false`. Rejection prepares a
  source-bound gap request.
- On `GAP_REQUIRED`, follow the official-first recovery section below.
- On `CAPABILITY_AUDIT_REQUIRED`, audit runtime version, file hashes, source and qualification before doing
  anything else. Never generate a same-name operator merely because provenance is unknown.
- On `BLOCKED`, stop and publish the stable reasons and minimal remediation. Do not weaken the scanner.

### 3. Generate the ordinary artifact with the configured model

First check local configuration. Run a network probe only when a real endpoint call is authorized and needed
as evidence:

```bash
python -m numpilot configure-llm --check
python -m numpilot configure-llm --probe
python -m numpilot doctor --backend npu
python -m numpilot migrate SOURCE --backend npu --out RUN_DIR
```

The migration command performs its own fresh qualified analysis and must fail closed on drift. Accept only a
write-once ordinary `script-migration-report.json` whose configured/requested/response model identity,
provider audit label, successful remote receipt, response hash, source hash, migrated hash and post-generation
AST audit all agree. `provider` is an audit label, not proof of a vendor. A successful AST audit proves only
that the remote artifact matches the allowed namespace rewrite; it does not turn local AST into the generator.

`migrate` does not execute either script. Its report must retain `execution_performed=false` and
`numerical_verification=not_run`.

### 4. Run explicit hash-bound numeric verification

Only after the user authorizes execution, select explicit outputs that both scripts assign and run:

```bash
python -m numpilot verify-script SOURCE \
  --migration-report RUN_DIR/script-migration-report.json \
  --output RESULT_NAME \
  --out RUN_DIR/script-verification-report.json
```

Repeat `--output` for additional numeric/bool values. Do not parse stdout or use pickle. The verifier must use
the report-sibling `migrated.py`, two isolated PIDs, bound interpreter/runtime identities, source/report/
artifact hashes, structure/shape/dtype/NaN/Inf checks and dtype-aware tolerances.

A PASS proves only the named values are numerically consistent in the recorded environments. It is not an OS
sandbox and it does not prove strict NPU execution, absence of fallback, profiler coverage, or speedup. Preserve
its fixed `strict_npu_verified=false` and `performance_valid=false` fields.

### 5. Run the separate strict-NPU resident performance gate

Only for a `migrated_strict_npu` artifact, an explicitly authorized execution, and a source with one declared
resident marker, run:

```bash
python -m numpilot benchmark-script SOURCE \
  --migration-report RUN_DIR/script-migration-report.json \
  --output RESULT_NAME \
  --out RUN_DIR/script-benchmark-report.json
```

Repeat `--output` where needed. The command rechecks source/report/artifact hashes and namespace-normalized
prefix/suffix AST equivalence, then uses separate CPU/NPU processes pinned to the same logical core. It must
observe CPU BLAS at one thread, retain warmups and raw samples, validate current runtime/profile/call counts,
synchronize before and after every NPU resident execution, perform one final D2H per named output, and pass the
same numeric contract before publishing speedup. Legacy `bench-cpu`/`bench-npu` cannot prove an ordinary
configured-model artifact.

The headline scope is the marker suffix only: steady device-resident compute, not kernel-only and not cold-start
end-to-end. Prefix/input preparation and final D2H are separate components. The reported warm-runtime component
sum is derived and must retain `directly_sampled_end_to_end=false`. If any hash, provenance, fallback, affinity,
thread, synchronization, call-count or numeric gate is unknown, require `performance_valid=false` and a null
speedup.

### 6. Independent audit

Ask `numpilot-evidence-auditor` to reproduce the allowed commands in a separately allocated evidence directory.
The auditor returns `PASS`, `FAIL`, or `BLOCKED`, evidence paths and hashes, exact reproduction commands, a
minimal counterexample, and explicit non-claims. The presence of `.codeartsdoer` files is not proof that a target
CodeArts account loaded or executed them; retain actual CodeArts product records when that claim is needed.

## Official-first gap recovery

Only `GAP_REQUIRED` may enter this lane. Bind every GapSpec to source SHA-256, callsite, semantic contract,
target SoC/CANN and the supported dtype/rank/layout range, then route in this order:

1. existing qualified AsNumpy/ACLNN public path;
2. installed CANN official binary/header evidence;
3. pinned `cann/ops-nn` official source;
4. minimal configured-model AsNumpy adapter over an exact official match;
5. only after both official-resolution layers produce auditable clean `NO_MATCH`, an isolated custom Ascend C
   draft generated by the configured model with the pinned, hash-verified CANNBot Skill bundle;
6. explicit CPU fallback or unsupported, neither of which may become a strict-NPU speed claim.

Probe failure, dirty checkout, fuzzy match, version/hash drift or missing evidence is not `NO_MATCH`; stop as
audit-required. Never silently write a shared OPP or system CANN location. Private build/install, public API,
real-NPU numerical, provenance and deny-fallback gates are required before qualification; benchmark/transfer
and appropriate profiler evidence are additionally required for performance claims.

Generated, reviewed, compiled, qualified, activated, rescanned and finally migrated/verified are distinct
states. The automatic `integrate -> qualify -> activate -> rescan -> final migrate/verify` executor is not yet
implemented. Use typed write-once stage receipts and report the actual stopping state instead of manually
copying artifacts and calling the loop complete.

## Required artifacts

- source, migrated source and diff, with SHA-256 bindings;
- qualified-analysis receipt with the five-state decision and runtime identity;
- configured-model identity and redacted remote receipt, plus AST audit;
- explicit script-verification receipt when execution was authorized;
- capability/backend provenance and fallback decision;
- benchmark receipt, raw samples and profiler/transfer artifacts only when those gates actually ran;
- build/test/stage receipts for any gap candidate;
- exact reproduction commands, failures, known limitations and prohibited extrapolations.

## Legacy advanced compatibility

`--mode contract` remains available only for explicit framework regression or advanced compatibility cases
whose source exposes `build_inputs() + forward()`. Its shim path and `bench-cpu`/`bench-npu` evidence apply only
to that contract artifact. Never substitute them for the ordinary configured-model competition chain.
