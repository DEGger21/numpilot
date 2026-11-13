# NumPilot framework recording demo

`large_numpy_program.py` is the user-facing terminal demo for the six-agent
migration workbench. It models a dense scientific feature contraction on one
FP32 data type: each large matrix projection is followed by a bounded `sin`
transform, repeated for five projection stages. This makes the NPU-friendly
work visible while keeping the source deterministic and easy to replay.

The source contains fourteen explicit NumPy call sites across the same admitted
family (`zeros`, `ones`, `add`, `eye`, `matmul`, `sin`): four setup calls and ten
repeated matrix/trigonometric stages. The ordinary workflow shows
Analyzer → Planner → Gap Filler → Code Migrator → Diagnostics → Reporter,
including the source path, run directory, backend, migration engine, performance
switch, recent events, and final evidence paths.

## Run the demo

From the repository root on the Ascend host:

```bash
V=${NUMPILOT_PYTHON:-python}
RUN=runs/demo-$(date +%Y%m%d-%H%M%S)
$V -m numpilot interactive-migrate   examples/demo/large_numpy_program.py   --backend npu --benchmark --out "$RUN" --yes --execute-source
```

The command is intentionally explicit about source execution and the output
boundary. It writes `workflow-report.json`, `workflow-progress.jsonl`, a
migration report, verification and benchmark receipts, and the migrated source
under the run directory. A failed qualification gate remains visible in the
report and is never presented as an NPU speedup.

## Configure your own model

Every user must create a private model configuration in the repository root before running the model-backed demo:

~~~bash
install -m 600 llm.example.json llm.json
${EDITOR:-vi} llm.json
python -m numpilot configure-llm --check
# Optional real connectivity check; this sends a model request and may incur cost.
python -m numpilot configure-llm --probe
~~~

Fill base_url, auth_token, and model with your own service values. The tracked llm.example.json remains blank, while llm.json is git-ignored and must not be copied from another user.

## API and evidence boundary

The static AsNumpy bundle currently has a strict FP32 2-D `matmul` profile; the
`sin + matmul` chain is the qualified scientific profile. Other public APIs may
be present in the component but are not automatically native-NPU qualified.
The report must therefore distinguish migrated API rewrites from a published
performance claim. If the runtime identity or profile is not bound, the UI
shows the blocked gate and preserves the evidence for diagnosis.

## Reproducibility lock (current release baseline)

The demo is reproducible only when the repository's immutable component bundle and the
matching runtime are used together. Do not install a floating `asnumpy` version from
PyPI or a Git branch. The current NumPilot baseline is:

| item | locked value |
|---|---|
| AsNumpy version | `0.3.1.dev70+g60d139c54.d20261108` |
| AsNumpy source commit | `60d139c54d9e181b8ae101dcbd1479d6b2a86439` |
| AsNumpy content tree | `f1c17d947a165c587a1b1d618d11d03fc5f5b183` |
| AsNumpy wheel SHA-256 | `d3ea03b2fd8c6eace56f532e2b30a917d63a138abe654c0020c14499898ed682` |
| CANNBot Skill bundle | `39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c` |
| CANNBot bundle SHA-256 | `da0ee3d6a12561da0d630125d7f4ae8602848dbd4c6f1b3f7150b26a57ead79f` |
| platform | Linux AArch64 / CPython 3.12 |
| runtime | CANN 9.2.0-beta.2 / NumPy 2.5.4 / Loguru 0.7.3 |

From a clean checkout on the target Ascend host:

```bash
mkdir -p runs/bootstrap
V=${NUMPILOT_PYTHON:-python}
COMPONENT="components/asnumpy"
RECEIPT="runs/bootstrap/receipt.json"

# Install exactly the wheel shipped by this checkout, then bind the receipt.
"$V" -m numpilot bootstrap --apply \
  --asnumpy-component "$COMPONENT" \
  --receipt "$RECEIPT" --json

# Verify the manifest, wheel hash, runtime identity, and CANN environment.
"$V" -m numpilot bootstrap --verify --json
"$V" -m numpilot doctor --backend npu --json
"$V" - <<'PY'
import asnumpy
print("asnumpy", getattr(asnumpy, "__version__", "unknown"))
PY
```

The demo command above then runs against the same `$V`. A successful reproduction must
keep the exact component manifest, wheel hash, CANN version, Python ABI, and input source
hash in the run directory. If any of these identities differ, treat the result as a new
benchmark rather than comparing it with the recorded demo. Older benchmark archives in
the repository are historical evidence and intentionally retain the version with which
they were recorded.

## Verified baseline result

The current locked baseline has a recorded single-core comparison for this exact source. The benchmark uses one CPU core and one Ascend910B3 card, with 3 warmup runs and 20 resident samples over the marker-delimited matmul + sin region:

| measurement | result |
|---|---:|
| resident CPU median | 427.217440 ms |
| resident NPU median | 10.229979 ms |
| median speedup | 41.761321x |
| conservative p10/p90 ratio | 40.095726x |
| numerical max absolute difference | 0 |

The headline speedup is explicitly for the steady device-resident region. One-time input preparation and final device-to-host transfer are recorded separately in the receipt, so a new run must keep the same timing scope before comparing numbers. The final receipt is preserved under the server run directory:
/home/ma-user/work/documents/numpilot-demo-evidence-20261113/script-benchmark-report-final.json
