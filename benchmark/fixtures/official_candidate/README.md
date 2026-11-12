# Official candidate fixture

`concatenate.py` is a deterministic, single-API fixture for the official
ACLNN candidate route. It is deliberately outside the immutable 150-case
benchmark scope. Its inputs contain no NumPy calls, so a resolver receipt can
bind exactly one `numpy.concatenate` GapSpec and the installed ACLNN Cat
candidate without conflating input construction with operator coverage.

The expected read-only commands on the Huawei runtime use an absolute source
path. This is required because GapSpec hashes include the canonical source
path:

```bash
cd /path/to/numpilot
FIXTURE="$PWD/benchmark/fixtures/official_candidate/concatenate.py"
python -m numpilot complete-gaps \
  "$FIXTURE" \
  --provider prepare \
  --target-soc Ascend910B3 \
  --cann-version 9.2.0-beta.2 \
  --out /tmp/numpilot-official-concat-prepare

python -m numpilot resolve-gaps \
  "$FIXTURE" \
  --target-soc Ascend910B3 \
  --cann-version 9.2.0-beta.2 \
  --out /tmp/numpilot-official-concat-resolution.json

python -m numpilot prepare-official-candidate \
  "$FIXTURE" \
  --official-resolution-report /tmp/numpilot-official-concat-resolution.json \
  --out /tmp/numpilot-official-concat-handoff.json
```

The expected resolution is `official_candidate_found` with route
`installed_cann_adapter_candidate`, ACLNN symbols
`aclnnCatGetWorkspaceSize` and `aclnnCat`, and all build/network/system-write
flags false. `complete-gaps --provider llm` must remain blocked until the
explicit official-candidate recovery/adapter hand-off is implemented.
