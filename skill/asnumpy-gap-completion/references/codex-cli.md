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
