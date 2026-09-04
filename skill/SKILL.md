---
name: numpilot-migration-gate
description: Migrate ordinary NumPy programs to AsNumpy/NPU with NumPilot's interactive terminal or an AI agent, including fail-closed API-gap routing, numeric checks, performance evidence, and reports.
---

# NumPilot Migration Gate

## Purpose and two primary entrypoints

This is NumPilot's repository-owned, open-source operating contract. It tells an interactive AI how
to use the framework; it is not a model or proof that a CodeArts account loaded it. Select one model
source before inspecting any model configuration:

- **Terminal/configured lane:** the user fills a private `llm.json`, starts `numpilot`, and drives a
  persistent terminal session like an ordinary development tool. The configured model supplies code
  interpretation, migration candidates, and eligible operator drafts. NumPilot supplies static
  analysis, receipt binding, NPU qualification, numeric comparison, benchmarking, and the final
  report. Low-level manual commands can still be used without a model, but the product-level
  `check llm` and `migrate` experience requires a usable configured endpoint.
- **AI/CodeArts-agent lane:** the user asks the interacting AI to read this Skill and the repository,
  then that same AI owns source interpretation, completion, operator authoring, integration work,
  and migration candidate generation. NumPilot remains the deterministic execution/evidence plane.
  This lane must not read or invoke `llm.json` or NumPilot's configured-model path.

A hybrid handoff is possible, but it is a composition of those two modes rather than a third model
source. Keep agent-authored and configured-model artifacts in different run directories with
separate provenance. A file below `.codeartsdoer/` is not evidence that CodeArts loaded or executed
it; retain the actual sidebar task/event record when that product-level claim matters.

## Product terminal flow

Use this framework-owned flow for any ordinary Python source; do not recreate it in a demo-only
script:

1. Run `numpilot` in a TTY. It opens the persistent NumPilot shell, renders the product mark, and
   lists `check llm`, `migrate`, `help`, and `exit`.
2. `check llm` sends one small, bounded request to the model selected by the private `llm.json` and
   succeeds only when a usable response is returned. Treat the command itself as authorization for
   that connectivity check. Show the model identity and a redacted pass/fail result; never print the
   token, endpoint, proxy URL, request body, or response body.
3. `migrate` asks for the target program's **absolute path**. Take a stable regular UTF-8 snapshot,
   bind its SHA-256/bytes/lines, and perform inert analysis. Display the NumPy APIs and callsites,
   their current AsNumpy status, and the exact gap or blocked reason without importing the source.
4. Ask the configured model for a bounded source interpretation, render it as
   `interpreted_unverified`, and then ask whether to proceed. Only an explicit `y` may authorize
   subsequent model generation, NPU work, and (when the prompt says so) the pinned CANNBot/CANN
   license for this invocation. Immediately before user scripts are executed, require the distinct
   exact input `execute`; `y` is not source-execution consent. Decline, EOF, or any other input at
   either gate leaves the affected later stages `not_run`.
5. With no gap, run the ordinary migration, named-output numeric comparison, and requested
   strict-NPU resident benchmark. With a gap, use the policy below; never continue merely because a
   plausible source file was generated.
6. Finish with the migrated-source path, execution/numeric status, measured shape/dtype/elements/
   bytes, benchmark facts only when `performance_valid=true`, every generated artifact path, and
   the machine-readable plus Markdown session reports. A measured speedup applies only to the exact
   recorded workload and timing boundary; do not infer a universal crossover size from one point.

### Registered `add_one` signature-gap route

NumPilot has one deliberately narrow, reusable completion path for the bundled API-gap example
`examples/add_one_migration_demo.py`. It recognizes only a direct `numpy.add(array, 1)` or
`numpy.add(array, 1.0)` call. Describe this accurately: AsNumpy has the qualified array-array
primitives, while this scalar-one call contract needs the run interface `asnumpy.add_one(x)` with
`float32`, shape-preserving semantics `y = x + 1`.

After the `y` gate, keep these two artifacts and claims separate:

1. Resolve official candidates first. Only an auditable clean miss opens custom generation.
2. Load and hash-check the pinned CANNBot Skill, display the full public context used by the request,
   and keep a periodic heartbeat visible while the real configured model works. Save the bounded
   Ascend C `Adds` fragment as `generated_unverified`. It is a design draft: it is not compiled,
   installed, activated, or used by the migrated program.
3. Independently materialize a run-scoped `asnumpy.add_one(x)` implementation from already qualified
   `asnumpy.ones_like` plus `asnumpy.add`. Qualify that composition in an isolated real-NPU process,
   including numeric cases, runtime identity, call counts, no-fallback evidence, and receipt hashes.
   Only a passing qualification may expose the interface for this run; it does not modify upstream
   AsNumpy or a shared OPP.
4. Prove the exact scalar-one lowering, obtain the separate `execute` confirmation, then migrate the
   normalized ordinary program and run the usual numeric and performance gates. The lowering uses
   the constituent `ones_like + add` calls rather than calling `asnumpy.add_one` directly, so the
   final report must keep interface integration false while reporting the independently qualified
   run-scoped interface and `cannbot_draft_activated=false`.

This special route does not generalize to arbitrary missing APIs. For every unregistered gap,
official candidate without full qualification, failed model call, or merely
`generated_unverified` custom draft, publish the stopping receipt and leave interface availability,
migration, verification, and benchmark false or `not_run` as applicable.

## Agent-mode option gate

Natural-language instructions such as “阅读这个 Skill，使用 agent 模式，不用 `llm.json` 里的内置
模型进行迁移” select the AI/CodeArts-agent lane immediately. Before any configuration read, `stat`,
probe, or configured-model command:

- do not open or pass `llm.json`, run `check llm`/`configure-llm`, pass `--provider llm`, inject a
  proxy for NumPilot, or silently fall back to the terminal/configured lane;
- read the repository and this Skill, discuss the source and possible designs with the user, and let
  the interacting AI itself author every interpretation, completion, operator candidate,
  integration change, and migration candidate in an isolated workspace;
- label the result `agent_authored_draft` and preserve agent/provider/model/task plus source/artifact
  hashes in a separate handoff; do not forge a configured-model response or migration report;
- run only deterministic NumPilot gates whose published input contract accepts that artifact. When
  no registered adapter can materialize the required hash-bound ordinary migration report, stop at
  `migrate=draft/stopped`, with `verify=not_run` and `benchmark=not_run`.

Changing model source requires a fresh run directory and new explicit authorization. Proxy settings
are optional site-specific environment details supplied by the user. This Skill requires only that
the selected model can return a usable response; it must never hard-code or infer a proxy address,
credential, or private endpoint.

## CodeArts interaction contract

When this skill is invoked in a sidebar, start with a short plan and a read-only analysis. Keep
these boundaries visible to the user:

1. **Read-only first.** Read the requested source and relevant repository docs, then run `analyze`
   in a new private run directory. Do not import or execute user code during analysis.
2. **Explicit approvals.** Ask immediately before each materially different action: a remote model
   request or network probe, user-script execution, NPU/build/install work, and any private OPP
   activation. A prior approval does not silently authorize later actions.
3. **Deterministic execution.** Use NumPilot CLI commands and preserve their JSON receipts. Do not
   replace a failed command with hand-edited code, an AST rewrite, an arbitrary shell pipeline, or
   a hidden model call.
4. **Report facts separately.** Return the five-state decision, stage status, artifact paths and
   hashes, model/runtime provenance, and explicit non-claims. In particular, numeric consistency,
   strict-NPU execution, and performance are different facts.
5. **Fail closed.** Stop at the first failed gate, retain its receipt, and tell the user the smallest
   safe next action. Never turn `unknown`, `skipped`, `fallback`, or `not_run` into `passed`.

For a compact no-`llm.json` sidebar prompt, use:

> 请阅读 `$numpilot-migration-gate` 和 NumPilot 仓库，使用 agent 模式，不用 `llm.json` 或
> NumPilot 内置模型。由你与我交互完成源码解读、缺口补全、算子生成、接入和迁移候选；随后调用
> 框架支持的确定性门禁并报告全部产物、哈希、数值与性能边界。没有适配器时如实停在
> `migrate=draft/stopped`，不要伪造后续回执。

## Skill identity and safe loading

Release tooling installs this directory under `share/numpilot/skills/numpilot-migration-gate/`. The
repository also exposes a human-visible copy at `skill/`, while CodeArts' source-checkout discovery
path remains `.codeartsdoer/skills/numpilot-migration-gate/`; both copies must remain byte-identical.
A loader or product shell should resolve one
regular, non-symlink `SKILL.md`, verify UTF-8 and a bounded size, and
record its SHA-256 in a private skill receipt. It may also record the hashes of `agents/openai.yaml`
and any references it displays. Never include private model configuration, tokens, endpoint URLs,
full response bodies, or unredacted environment dumps in that receipt or in a public evidence set.

For the terminal display/receipt schema and redaction rules, read
[references/terminal-contract.md](references/terminal-contract.md). For sidebar prompts, status
rendering, and the distinction between CodeArts control-plane evidence and NumPilot execution
evidence, read [references/codearts-sidebar.md](references/codearts-sidebar.md).

## Trust boundary

The default input is an ordinary, self-contained Python file with standard
`import numpy as np`. Do not require `build_inputs()` or `forward()`. Static admission, candidate
generation (manual, CodeArts-agent, or explicitly configured model), explicit execution, NPU
provenance, and performance are separate claims with separate receipts.

When an artifact requires configured-model provenance, use the endpoint explicitly selected in the private
configuration through the supported OpenAI-compatible protocol; the exact model name remains bound by
configuration and receipt. A manual or CodeArts-agent draft is not silently upgraded to that claim. Local AST
logic may construct and check the exact expected namespace rewrite, but it must never generate that artifact or
replace a failed model call. Do not use `--engine ast`, `--mode auto`, or `--mode contract` when configured-model
provenance is required.

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

Read `README.md`, `docs/architecture.md`, `docs/codearts-integration.md`, and the relevant capability/provenance
records. In CodeArts, optionally ask `numpilot-analyzer` for a read-only review;
in a terminal/manual run, perform the same review locally. Then have the deterministic execution plane
publish a write-once receipt:

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

- On `DIRECT`, continue according to the selected entry lane: an explicitly configured terminal
  sub-run may call the model, while manual, CodeArts-agent, and hybrid drafts proceed to the
  deterministic candidate checks without reading configuration implicitly.
- On `FALLBACK_CONSENT`, pause for explicit acceptance or rejection. Acceptance may produce a compatibility
  artifact only; it is always `strict_npu=false` and `performance_valid=false`. Rejection prepares a
  source-bound gap request.
- On `GAP_REQUIRED`, follow the official-first recovery section below.
- On `CAPABILITY_AUDIT_REQUIRED`, audit runtime version, file hashes, source and qualification before doing
  anything else. Never generate a same-name operator merely because provenance is unknown.
- On `BLOCKED`, stop and publish the stable reasons and minimal remediation. The only current
  exception is an exact, source-bound match by the registered scalar-one recognizer described above;
  it may enter that narrow route without weakening or relabelling the ordinary scanner result.

### 3. Produce and admit the ordinary candidate

The candidate authoring boundary depends on the selected lane. In the terminal/manual lane, the user
may edit or supply the candidate directly; in the CodeArts-agent lane, the interactive model authors it
after reading the repository and Skill; in a hybrid run, either of those drafts is handed to the
terminal. Mark non-configured candidates `agent_authored_draft` or `user_authored_draft` and retain
their source/artifact hashes. Do not create a configured-model receipt for them. There is currently no
generic adapter that turns an arbitrary hand-authored ordinary file into the schema-2
`script-migration-report.json` consumed by `verify-script`; absent a registered adapter, record the
candidate hash and stop rather than pretending that deterministic verification can run.

For the optional configured-model terminal sub-run, first check local configuration. Run a network
probe only when a real endpoint call is authorized and needed as evidence:

```bash
python -m numpilot configure-llm --check
python -m numpilot configure-llm --probe
python -m numpilot doctor --backend npu
python -m numpilot migrate SOURCE --backend npu --out RUN_DIR
```

For that configured sub-run, `migrate` performs its own fresh qualified analysis and must fail closed on
drift. Accept only a write-once ordinary `script-migration-report.json` whose configured/requested/response
model identity, provider audit label, successful remote receipt, response hash, source hash, migrated hash and
post-generation AST audit all agree. `provider` is an audit label, not proof of a vendor. A successful AST
audit proves only that the remote artifact matches the allowed namespace rewrite; it does not turn local AST
into the generator. Manual, CodeArts-agent, and hybrid drafts instead require an adapter/handoff contract
that binds their author, source hash, artifact hash, and lane before deterministic verification.

`migrate` does not execute either script. Its configured report must retain `execution_performed=false` and
`numerical_verification=not_run`.

### 4. Run explicit hash-bound numeric verification

Only after the user authorizes execution, select explicit outputs that both scripts assign and run.
For a manual or CodeArts-agent candidate, first check whether a registered adapter can materialize a
hash-bound ordinary migration report. If it cannot, record `migrate=draft/stopped`, set
`verify=not_run` and `benchmark=not_run`, and stop at this boundary; never hand-edit a
configured-model report to disguise its origin. Only an adapter-accepted report may use:

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

Only for a `migrated_strict_npu` artifact whose lane/author provenance is bound, an explicitly authorized
execution, and a source with one declared resident marker, run:

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
configured-model artifact, and an unbound manual/agent draft cannot prove strict-NPU performance.

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

Only `GAP_REQUIRED` or the exact registered scalar-one signature may enter this lane. The latter keeps
the ordinary scanner's original `BLOCKED` fact and records the separate registered-route decision. Bind
every GapSpec to source SHA-256, callsite, semantic contract, target SoC/CANN and the supported
dtype/rank/layout range, then route in this order:

1. existing qualified AsNumpy/ACLNN public path;
2. installed CANN official binary/header evidence;
3. pinned `cann/ops-nn` official source;
4. minimal adapter over an exact official match (configured-model or CodeArts-agent according to the
   selected lane);
5. only after both official-resolution layers produce auditable clean `NO_MATCH`, an isolated custom Ascend C
   draft generated by the explicitly authorized boundary model (configured model or CodeArts agent) with the
   pinned, hash-verified CANNBot Skill bundle. Label a non-configured draft `agent_authored_draft` and keep
   its CodeArts task/event evidence separate;
6. explicit CPU fallback or unsupported, neither of which may become a strict-NPU speed claim.

Probe failure, dirty checkout, fuzzy match, version/hash drift or missing evidence is not `NO_MATCH`; stop as
audit-required. Never silently write a shared OPP or system CANN location. Private build/install, public API,
real-NPU numerical, provenance and deny-fallback gates are required before qualification; benchmark/transfer
and appropriate profiler evidence are additionally required for performance claims.

Generated, reviewed, compiled, qualified, activated, rescanned and finally migrated/verified are distinct
states. The generic automatic `integrate -> qualify -> activate -> rescan -> final migrate/verify` executor is
not implemented. The registered scalar-one route above is a narrower exception: it qualifies a run-scoped
composition of existing AsNumpy APIs, while its CANNBot Ascend C draft stays uncompiled and unactivated. For
all other gaps, use typed write-once stage receipts and report the actual stopping state instead of manually
copying artifacts and calling the loop complete.

## Required artifacts

- source, migrated source and diff, with SHA-256 bindings (migrated/diff are conditional on a candidate
  being admitted);
- qualified-analysis receipt with the five-state decision and runtime identity;
- configured-model identity and redacted remote receipt, plus AST audit (when the configured lane was used);
- manual/CodeArts-agent/hybrid author label, handoff and task/event reference when no configured model was used;
- explicit script-verification receipt when execution was authorized and an adapter-accepted report exists;
- capability/backend provenance and fallback decision;
- benchmark receipt, raw samples and profiler/transfer artifacts only when those gates actually ran;
- build/test/stage receipts for any gap candidate;
- final private `migration-report.json` and `migration-report.md`, plus a generated-file inventory;
- exact reproduction commands, failures, known limitations and prohibited extrapolations.

## Legacy advanced compatibility

`--mode contract` remains available only for explicit framework regression or advanced compatibility cases
whose source exposes `build_inputs() + forward()`. Its shim path and `bench-cpu`/`bench-npu` evidence apply only
to that contract artifact. Never substitute them for the ordinary lane-specific workflow or use them
to upgrade a manual/agent draft into configured-model evidence.
