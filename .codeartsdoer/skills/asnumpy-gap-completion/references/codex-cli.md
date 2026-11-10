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
