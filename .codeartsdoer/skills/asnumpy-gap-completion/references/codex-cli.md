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
