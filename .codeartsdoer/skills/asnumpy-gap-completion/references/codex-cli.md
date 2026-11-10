# Codex CLI / local model hand-off

A host agent can submit one locally generated bundle without giving NumPilot access to `llm.json`:

```bash
python -m numpilot complete-gaps path/to/script.py \
  --provider injected \
  --response-file /absolute/path/model-response.json \
  --target-soc Ascend910B3 \
  --cann-version 9.2.0-beta.2 \
  --out /absolute/path/run
```

`model-response.json` is a UTF-8 JSON object with `summary` and `files`, or a
`NUMPILOT_BUNDLE_V1` line bundle. The CLI accepts only a local regular file no
larger than 3 MiB. It does not read `llm.json`, access the network, compile,
install, or activate the candidate. The resulting artifact is recorded as
`mode=injected_test_double`, `provider=codex-self`, `model=codex-local`, and
`status=generated_unverified`; the normal GapSpec, AsNumpy manifest, ACLNN
two-stage, explicit-handle-dlsym, provenance, and artifact-store gates still
apply.

For an ASC candidate, put the `.asc` file under an allowed overlay directory
such as `csrc/`, include one `asnumpy_adapter/*.json` manifest, and declare
`asnumpy.adapter.manifest.v1` with `explicit-handle-dlsym`. A passing injected
run is only a structural hand-off; numerical, build, NPU, performance,
qualification, and activation gates remain pending.


## Official AsNumpy adapter handoff

For an official existing-module candidate, NumPilot owns preparation and
validation while Codex acts only as a local generation agent:

```bash
python -m numpilot prepare-asnumpy-adapter-handoff \
  --candidate /abs/path/operator-candidate.json \
  --recovery-state-dir /abs/path/recovery \
  --official-context-gate /abs/path/official-context-gate.json \
  --asnumpy-reference /abs/path/asnumpy \
  --out /abs/path/codex-handoff
```

The command does not read `llm.json`, access the network, build, import, or
modify AsNumpy. It writes `handoff.json`, hash-bound `context/`, reviewed
`windows/`, per-file `prompts/`, and an empty `responses/` directory. In a
separate Codex CLI session, read `README.md` and each prompt, then write exactly
one `NUMPILOT_WINDOW_REPLACE_V1` response to
`responses/<target-path>.response`.

Before asking the external Codex session to write files, or before running
validation, inspect the hand-off directory with the read-only status command:

```bash
python -m numpilot inspect-asnumpy-adapter-handoff \
  --handoff-dir /abs/path/codex-handoff \
  --response-dir /abs/path/codex-handoff/responses
```

Use `--json` when another agent or a script needs a machine-readable result.
The command checks only the hash-bound expected response paths. Each response
must be a regular, non-empty UTF-8 file no larger than 2 MiB; symlinks, empty
files, unreadable files, and over-sized files are reported in
`invalid_responses`. It does not read `llm.json`, use the network, import
AsNumpy, build, execute, activate, or write RecoveryStore state.

Interpret the result as follows:

- `status=waiting_for_agent`: at least one expected response is missing or an
  existing response is invalid. The generation session still needs to write or
  repair `responses/<target-path>.response`; this is an expected waiting state,
  not a failed operator generation.
- `status=ready_for_validation`: every expected response is present and passes
  the local file checks. Run `validate-asnumpy-adapter-handoff` next; it is the
  command that parses and hash-validates the replacement protocol.

The status command never replaces validation. A response can be locally ready
and still fail the protocol, source-context, or later AsNumpy gates.

NumPilot then consumes the local responses:

```bash
python -m numpilot validate-asnumpy-adapter-handoff \
  --handoff-dir /abs/path/codex-handoff \
  --response-dir /abs/path/codex-handoff/responses \
  --out /abs/path/codex-adapter
```

A successful result is `schema_version=5`,
`generation_mode=injected_test_double`, `provider=codex-self`, and
`actual_remote_call=false`. It is still only `generated_unverified`; build,
import, NumPy oracle, real NPU, performance, qualification, and activation
remain NumPilot gates.

### Consume the validated local adapter

After the status command reports `ready_for_validation`, consume the package in
this order. These commands use only local artifacts and do not call `llm.json`:

```bash
python -m numpilot validate-asnumpy-adapter-handoff \
  --handoff-dir /abs/path/codex-handoff \
  --response-dir /abs/path/codex-handoff/responses \
  --out /abs/path/codex-adapter

python -m numpilot recover-gaps ingest-codex-adapter \
  --state-dir /abs/path/recovery \
  --candidate /abs/path/operator-candidate.json \
  --adapter-output /abs/path/codex-adapter \
  --json

python -m numpilot recover-gaps preflight-integrate-candidate \
  --state-dir /abs/path/recovery \
  --candidate /abs/path/operator-candidate.json \
  --adapter-output /abs/path/codex-adapter \
  --asnumpy-reference /abs/path/asnumpy \
  --python /abs/path/python \
  --json
```

`ingest-codex-adapter` accepts only the schema-5 local profile and advances
RecoveryStore to `ADAPTER_CANDIDATE`; it does not build or import AsNumpy. The
preflight is read-only. A receipt-only local profile is expected to return
`status=blocked`, `code=authorization_missing`, and
`next_action=authorize_full_local_recovery` until the operator explicitly
provides the full-local-recovery policy. Only after a `status=ready`, `code=ready_for_integration`
preflight may the operator run `integrate-candidate` with a private absolute
`--out` directory. Integration then remains subject to build, import, NumPy
oracle, real NPU, performance, qualification, and activation gates.

Do not skip a failed or blocked step, do not copy a response directly into the
AsNumpy source tree, and do not convert a local Codex receipt into a remote
DeepSeek receipt. Every command above is replayable from the hash-bound handoff
and candidate paths.


The official materializer now accepts `--profile full-local-recovery` for a
**new** run, when local builds/execution are authorized. Use
`materialize-official-candidate-recovery --help` for the bound handoff,
preflight, resolution, context gate, source, target and runtime-identity
arguments. An existing run cannot change its policy: create a new state and
candidate-output directory from the same verified inputs, then ingest the
validated adapter there. The runtime identity must come from an actual probe,
not a test fixture. `audit_required` remains unqualified even if integration
passes. This explicit local profile does not itself call a remote model or
activate a candidate.


### Qualify an integrated official concatenate adapter

The real Codex route now continues through an independently measured CANN
kernel profile, the official qualification producer, and the original I006
AST/strict-NPU migration. See
`docs/codex-adapter-walkthrough.zh-CN.md` for the three runnable commands and
runtime selection. Qualification keeps the native official handoff schema and
requires transaction-bound evidence. Its explicit overlay admits only its
reviewed API/profile; the global runtime can remain `audit_required`.

The integration RecoveryStore record, published qualification overlay, and
sample migration receipt are separate products. Do not manually edit the
store to claim activation, increment the 150-case suite from a direct smoke,
or turn an independent profiler count into a count for another execution.
