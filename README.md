<div align="center">

<h1><img src="docs/assets/brand/numpilot-wordmark.png" alt="NumPilot" width="560"></h1>

### End-to-end NumPy → AsNumpy migration for Ascend NPUs

<p>
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#results">Results</a> ·
  <a href="#contributing--license">Contributing</a>
</p>

<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.3.0a1.dev0-6D5CE7?style=flat-square">
  <img alt="Release stage" src="https://img.shields.io/badge/status-alpha-F59E0B?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/Python-as%20supported%20by%20AsNumpy-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white">
  <img alt="NPU" src="https://img.shields.io/badge/NPU-Ascend%20(runtime--detected)-008AA3?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-2EA44F?style=flat-square">
</p>

</div>

## Overview

NumPilot is an end-to-end migration framework that turns complete NumPy programs into complete
AsNumpy programs for Ascend NPUs, with every stage producing auditable receipts.

[AsNumpy](https://github.com/asnumpy/asnumpy) is the NumPy-compatible tensor library for Ascend
NPUs, developed by the AISS team at Harbin Institute of Technology led by Prof. Tonghua Su.

## Quick start

NumPilot has two supported installation lanes. Use the frozen component lane on an Ascend host when you want to reproduce the published demo. Use the CPU/shim lane for source development without CANN.

**Frozen Ascend lane requirements:** Linux AArch64, CPython 3.12, an Ascend device, CANN 9.0 or newer, and a user-owned virtual-environment directory. The recorded demo result below uses the audited CANN 9.2.0-beta.2 profile.

~~~bash
git clone --branch develop --depth 1 https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot
mkdir -p "$HOME/.local/venvs"
./install.sh --allow-network \
  --python "$(command -v python3.12)" \
  --asnumpy-component "$PWD/components/asnumpy" \
  --cann-env /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh \
  --venv "$HOME/.local/venvs/numpilot" \
  --persistent-root "$HOME/.local/share/numpilot"
. "$HOME/.local/share/numpilot/activate-numpilot.sh"
python -m numpilot bootstrap --verify --json
python -m numpilot doctor --backend npu --json
~~~

The installer installs the exact wheel shipped in components/asnumpy, writes a hash-bound bootstrap receipt, and checks the CANN/NPU runtime. It never downloads a floating AsNumpy version. Do not copy an existing virtual environment between machines.

**CPU/shim lane:**

~~~bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m numpilot doctor --backend shim
python -m pytest -q -p no:cacheprovider tests/test_install_script.py tests/test_bootstrap.py
~~~

Migrate and verify a complete program:

~~~bash
numpilot analyze program.py
numpilot migrate program.py --out runs/demo
numpilot verify program.py \
  --migration-report runs/demo/migration/script-migration-report.json \
  --output result --out runs/demo/verification.json
~~~

For configured-model gap completion, copy the blank template and fill in your own endpoint. A model call is only made by the configured-model lane; the ordinary analyzer, verifier, benchmark, and reporter use deterministic gates and receipts:

~~~bash
install -m 600 llm.example.json llm.json
# edit llm.json: base_url, auth_token, model
python -m numpilot configure-llm --check
~~~

The complete release procedure is in [Installation](docs/installation.md).

### Interactive workbench

Start the product shell from the activated environment:

~~~bash
numpilot
~~~

The startup page shows the NumPilot logo, a model status bar, and five choices:
new migration, load migration memory, check model, environment self-check, and
exit. The status starts as `未检查`; run `/modelcheck` (or choose option 3) to
send one real lightweight probe to `llm.json`, then the bar shows `模型可用` or
`模型不可用`. Model checks are always explicit; startup does not call the model.

| Command | Purpose |
| --- | --- |
| `/help` | Interaction help, status phases, and artifact layout |
| `/new` | Prompt for an absolute NumPy path and create a fresh session |
| `/resume` | Load a previous migration memory |
| `/readreport` | Render the latest `migration-report.md` in the terminal |
| `/modelcheck` | Manually probe the configured model |
| `/doctor` | Check CANN, NPU, Python, and AsNumpy |
| `/status` | Show the current model status |
| `/migrate /abs/x.py` | Start migration with a known absolute path |
| `/exit` | Leave NumPilot |

Every migration creates `runs/demo-<timestamp>/` and writes the full session
there: `source/<program>.py` (a copy of the user's source),
`migration-plan.md/json` (Analyzer/Planner facts and the model plan),
`migration-report.md/json`, and `workflow/` receipts and logs. A source without
NumPy imports or calls is rejected before any model request. Agent status is
monotonic: a completed role never jumps back to running, and Diagnostics and
Reporter finish only after their model diagnosis/report step.

### Run the demo

Run from the repository root after activating the NumPilot environment. Configure
`llm.json` first:

~~~bash
install -m 600 llm.example.json llm.json
${EDITOR:-vi} llm.json   # base_url, auth_token, model
python -m numpilot configure-llm --check
~~~

Then start the workbench:

~~~bash
numpilot
~~~

Inside the shell:

~~~text
/modelcheck
/new
/path/to/numpilot/examples/demo/large_numpy_program.py
y
/readreport
/resume
/exit
~~~

Use the absolute path to the checkout's `examples/demo/large_numpy_program.py`.
`/new` creates a fresh `runs/demo-<timestamp>/`; `y` confirms migration;
`/readreport` renders the final Markdown report; `/resume` lists the saved
migration memory.

For non-interactive automation, the same demo can be started directly:

~~~bash
python -m numpilot interactive-migrate "$PWD/examples/demo/large_numpy_program.py" \
  --backend npu --benchmark --out "runs/demo-$(date +%Y%m%d-%H%M%S)" --yes --execute-source
~~~

`llm.example.json` stays blank; `llm.json` is private and git-ignored. Every user
must create and configure their own copy; do not reuse another user's token or
endpoint.

## Architecture

<p align="center">
  <img src="docs/assets/architecture/numpilot-architecture-en.png" alt="NumPilot architecture" width="100%">
</p>

The pipeline runs six roles — Analyzer, Planner, API Gap Filler, Code Migrator, Diagnostics,
Migration Reporter — across three stages: analyze, migrate, verify. Deterministic gates own
parsing, capability resolution, builds, numerical comparison and receipts; a model may propose
code, but it cannot upgrade an unverified claim.

## Results

### 1. Vitamin C: real quantum-chemistry grid-density chain

The example uses a real Vitamin C molecule (20 atoms, 92 electrons, 208 basis functions, 81,936 DFT
grid points) and runs the chain `dm.T @ AO → rho → weighted → electron_count`.

| Scope | CPU single core | NPU resident | NPU warm end-to-end |
| --- | ---: | ---: | ---: |
| Full grid-density chain | 341.750 ms | 13.157 ms (**25.98x**) | 17.419 ms (**19.62x**) |

The SCF (~294.7 s) stays on CPU; the speedup covers the grid-density sub-stage only. Inputs are
float32, `projected` max absolute error is 9.54e-6, `rho` is 1.83e-4, and the electron count
matches the CPU same-form result exactly. This example is a hand-written migration using
NumPilot's audit runtime (`model_called=false`), not a model-generated operator.

Details: [Vitamin C example](examples/scientific/gpu4pyscf_vitamin_c/README.md).

For a terminal recording of the framework-level six-agent workbench, see [the large NumPy demo](examples/demo/README.md). It shows the live agent panel, model migration receipt, numerical verification, and the strict boundary around NPU performance claims.

The demo is pinned to the release-adjacent AsNumpy component in [components/asnumpy/](components/asnumpy/); its wheel, source snapshot, manifest, and SBOM are the reproducibility lock.

### 2. 71 programs without API gaps

71 complete NumPy programs using only APIs already covered by AsNumpy; 66 have CPU/NPU
measurements, with a median speedup of 11.06x. The four fastest by speedup:

| Program | Workload | CPU / NPU (ms) | Speedup |
| --- | --- | ---: | ---: |
| `A009_matrix_chain_2048` | two 2048 × 2048 float32 matmuls; 34.4 GFLOP | 1225.11 / 2.00 | **612.41x** |
| `A007_matrix_product_2048` | 2048 × 2048 float32 matmul; 17.2 GFLOP | 608.88 / 1.68 | **361.58x** |
| `B015_fortran_matmul_2048` | 2048 × 2048 Fortran-ordered float32 matmul | 634.30 / 1.79 | **355.01x** |
| `H015_real_vecdot_16384x256` | 16,384 × 256 float32; vecdot over both axes | 91.98 / 0.56 | **164.79x** |

Full table: [71-case migration benchmark](examples/scaled_benchmark_20261112/BENCHMARK_TABLE.md).

> **Historical evidence note.** The 2026-11-12 71-case archive is retained as immutable evidence with the runtime recorded in its own README. Its AsNumpy/CANN versions are not the current component baseline. Reproduce the current release-adjacent path with `examples/demo/`, `examples/scientific/`, and `components/asnumpy/`.

### 3. Programs with real API gaps

Six complete programs whose NumPy call is a real API gap. Each operator was generated by the
configured model (`llm.json`), built, deployed, numerically validated and admitted before the
whole program was migrated through NumPilot's six-role pipeline. The four positive speedups:

| Program | Gap API | Workload | End-to-end time | CPU / NPU (ms) | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| `std` | `np.std` | 2048 × 2048 float32; std over axis=1 | 90.9 min | 15.96 / 0.3718 | **42.92x** |
| `tril` | `np.tril` | 2048 × 2048 float32; lower triangle | 38.4 min | 14.16 / 1.649 | **8.59x** |
| `ediff1d` | `np.ediff1d` | 2048 × 2048 float32 flattened; 4,194,303 differences | 60.7 min | 4.48 / 1.664 | **2.69x** |
| `pad` | `np.pad` | 2048 × 2048 float32; constant pad | 47.0 min | 4.117 / 1.724 | **2.39x** |

End-to-end time starts at the real operator-generation call and includes build, repair, deployment,
numerical validation, adapter build and tests, complete-program migration, and the CPU/NPU
benchmark. The migration command itself takes 6–8 s.

Five of the six complete-program migrations pass whole-program numerical comparison. The `tril`
NumPilot migration receipt is retained as a failure (`runtime target snapshot is invalid`); its
independent strict-NPU benchmark passed at 8.59x, but that benchmark does not certify a complete
NumPilot migration. All 84 adapter tests pass. The full table also records `flip` (0.0042x, CPU view baseline) and `argpartition` (0.03x, 512-element
launch-bound) as correctness cases.

Full table: [gap-operator migration archive](examples/gap_archive_20261112/BENCHMARK_TABLE.md).

## Contributing & License

Issues and pull requests are welcome. Open an issue on the repository, or contact
[li-xing@stu.hit.edu.cn](mailto:li-xing@stu.hit.edu.cn). See [CONTRIBUTING.md](CONTRIBUTING.md)
and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

NumPilot is released under the [Apache-2.0 License](LICENSE).
