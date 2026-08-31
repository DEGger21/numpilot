<div align="center">

<h1><img src="docs/assets/brand/numpilot-wordmark.png" alt="NumPilot" width="560"></h1>

### From NumPy programs to verified execution on Huawei Ascend NPU

<p>
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="docs/installation.md">Installation</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="evidence/README.md">Evidence</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.3.0a1.dev0-6D5CE7?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/status-development--snapshot-F59E0B?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white">
  <img alt="Release branch" src="https://img.shields.io/badge/release_branch-main-0F766E?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-2EA44F?style=flat-square">
</p>

<p>
  <a href="docs/installation.md"><img alt="Quick install" src="https://img.shields.io/badge/Quick_install-Open_guide-5B4EE8?style=for-the-badge"></a>
  <a href="docs/numpilot-en-v6.svg"><img alt="View architecture" src="https://img.shields.io/badge/Architecture-View_diagram-008AA3?style=for-the-badge"></a>
</p>

</div>

> [!IMPORTANT]
> NumPilot is an Alpha development snapshot, not a published release candidate. A successful installer or
> `doctor` run proves basic NPU availability only. Strict-NPU and performance claims are valid solely for the
> exact runtime profile and hash-bound receipts that produced them.

**NumPilot's goal is to automatically migrate every NumPy API in an input program to a functionally equivalent
NPU-backed implementation.** By preserving program semantics and numerical results while taking advantage of
the NPU's massively parallel compute capacity, NumPilot is designed to accelerate large-scale data processing.

The current Alpha snapshot supports a statically analyzable subset of ordinary, single-file NumPy programs.
Inputs use a top-level `import numpy as np` and must pass a fail-closed, whole-program capability gate before
migration. The installed AsNumpy/CANN/SoC runtime, model response, migrated AST, numerical result, and optional
performance claim are all checked against explicit evidence.

> **The model proposes a candidate. Deterministic evidence decides whether it may be adopted.**

<table align="center">
  <tr>
    <td align="center"><img src="docs/assets/brand/asnumpy-mark.png" alt="AsNumpy" height="54"><br><sub>AsNumpy</sub></td>
    <td align="center"><img src="docs/assets/brand/cann-wordmark.png" alt="CANN" height="34"><br><sub>CANN / ACLNN</sub></td>
    <td align="center"><img src="docs/assets/brand/ascend-wordmark.png" alt="Ascend" height="38"><br><sub>Ascend NPU</sub></td>
    <td align="center"><img src="docs/assets/brand/codearts-icon.png" alt="CodeArts" height="54"><br><sub>Optional CodeArts collaboration</sub></td>
  </tr>
</table>

## Architecture: from NumPy input to verified NPU execution

<p align="center">
  <a href="docs/numpilot-en-v6.svg">
    <img src="docs/numpilot-en-v6.svg" alt="NumPilot architecture: NumPy analysis, AsNumpy capability routing, gap completion, and verified Ascend NPU execution" width="100%">
  </a>
</p>

<p align="center">
  <sub>
    Open the <a href="docs/numpilot-en-v6.svg">full-size English diagram</a> ·
    <a href="docs/numpilot-zh-v6.svg">中文架构图</a> ·
    <a href="docs/numpilot-en.pptx">English PPT</a> ·
    <a href="docs/assets/architecture/numpilot-architecture-en.svg">machine-auditable boundary map</a>
  </sub>
</p>

The v6 diagram distinguishes current primitives with solid paths from target automation with dashed paths. The
text summary below preserves the same meaning when SVG rendering or path-based text is inaccessible.

### How to read the architecture

| Stage | What NumPilot does | Gate or output |
|---|---|---|
| 1. Unclassified workload | Parses the complete program, extracts NumPy callsites, signatures, and bounded data flow | No API is labelled supported or missing at intake |
| 2. Capability check | Matches every callsite against the exact AsNumpy/CANN/SoC profile and records an independent decision | The strictest per-call result becomes the fail-closed workload result |
| 3. Whole-file migration | If every call is eligible, the configured model rewrites the complete file and NumPilot applies exact AST and response checks | A migration candidate, never an automatically trusted result |
| 4. Gap completion | If an API is missing, NumPilot looks for reusable Huawei official operators first; only after clean official misses may CANNBot Skills guide the configured model to draft a custom operator | Both routes converge on the same AsNumpy API integration boundary |
| 5. Qualification and feedback | Integration candidates pass artifact, build, unit, numerical, public-API, and real-NPU gates, then the original workload is rescanned | Only a fully eligible rescan may return to migration and verification |

<details>
<summary><strong>Text-only architecture flow and current route boundaries</strong></summary>

The ordinary, already-supported route is:

```text
unclassified NumPy source
  → whole-file AST + bounded data-flow analysis
  → per-call, runtime-bound capability decisions
  → fail-closed whole-workload result
  → configured-model candidate rewrite
  → model receipt + exact AST gate
  → independent numerical verification
  → optional profile-bound strict-NPU benchmark
  → write-once evidence receipts
```

The API-gap route shown in the lower half of the diagram is:

```text
classified API gap
  → A. reuse an installed ACLNN candidate or audit a pinned ops-nn source candidate
  → B. after two clean misses, use pinned CANNBot Skills + the configured model
       to draft a custom operator
  → integrate through the AsNumpy public API boundary
  → qualify → activate capability → rescan the original workload
  → migrate only after every callsite is eligible
```

Route outputs remain deliberately distinct. An installed official-binary discovery can reach
`CANDIDATE_READY`; only the reviewed installed-ACLNN MatMul slice currently reaches `ADAPTER_CANDIDATE`; pinned
`ops-nn` currently provides `SOURCE_EVIDENCE_ONLY`; and the CANNBot/model route produces an
`UNVERIFIED_OPERATOR_DRAFT`. Generic integration, qualification, activation, rescan, mixed-route orchestration,
and rollback are still target paths rather than an end-to-end implemented loop.

</details>

> **Core invariant:** the input workload is never pre-classified, a gap is never silently bypassed, and a newly
> completed API must be integrated into AsNumpy and survive a rescan before migration can continue.

## What is implemented

| Area | Current state | Boundary |
|---|---|---|
| Ordinary NumPy analysis | Implemented | Static whole-file analysis; it does not execute source or call a model |
| Five-state capability decision | Implemented | Bound to the installed AsNumpy/CANN/SoC identity and audit evidence |
| Configured-model rewrite | Implemented | OpenAI Chat Completions-compatible endpoint; exact model/response/AST/hash checks fail closed |
| Numerical verification | Implemented | Original and migrated programs run in separate `python -I` workers; this is not an OS sandbox |
| Strict-NPU resident benchmark | Implemented, scoped | Requires an exact qualified profile; it is not a universal or cold-E2E claim |
| Official operator discovery | Implemented | Checks installed ACLNN, then pinned local `ops-nn` read-only source evidence; discovery does not build or integrate |
| Custom-operator drafting | Partial | After two clean misses, CANNBot Skills constrain the configured model; the resulting project remains unverified |
| Generic gap completion | Partial | Official binary reaches `CANDIDATE_READY`; only reviewed MatMul reaches `ADAPTER_CANDIDATE`; qualification and activation remain planned |
| CodeArts collaboration | Optional integration | Repository roles/skills exist; the actual control plane must be verified in the target account |

## Quick start

### CPU/shim development

This path lets contributors inspect the CLI and run a shim smoke check without CANN or an NPU:

```bash
git clone --branch main --depth 1 https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m numpilot --help
python -m numpilot doctor --backend shim
```

The complete suite is a release-matrix gate, not a portable quick-start check. It must run in the pinned test
environment with inherited CANN `PYTHONPATH` and proxy variables isolated; see [Contributing](CONTRIBUTING.md).

### NPU source-first installation

The target machine must already have a compatible Ascend driver, CANN, OPP, Python development headers,
Git, CMake, Make or Ninja, GCC 11.2+, and the system `fmt`/`spdlog` CMake development packages. NumPilot never
runs `sudo` or edits shell startup files.

```bash
./install.sh --allow-network \
  --cann-env /usr/local/Ascend/cann-9.0.0/set_env.sh \
  --venv /absolute/user-owned/numpilot-venv \
  --persistent-root /absolute/user-owned/numpilot-runtime

. /absolute/user-owned/numpilot-runtime/activate-numpilot.sh
python -m numpilot bootstrap --verify --json
python -m numpilot doctor --backend npu
```

The direct parent of `--venv` must already exist, be owned by the current user, and not be group/world writable.
Use paths owned exclusively by your account. `--allow-network` is explicit consent for dependency installation
and fetching the pinned AsNumpy source when it is not cached. The complete dependency matrix, CANNBot option,
post-install receipts, strict-profile check, troubleshooting, and safe cleanup are in the
**[installation guide](docs/installation.md)**.

Do not use `pip install numpilot` for this snapshot: there is no verified public package release for the current
source tree.

## Configure the model

```bash
python -m numpilot configure-llm --init
# Edit the private llm.json path printed by the command.
python -m numpilot configure-llm --check   # local validation; no network
python -m numpilot configure-llm --probe   # one real request; may incur cost
```

The configuration directory must be private (`0700`) and `llm.json` must be `0600`. Tokens, private endpoints,
proxy URLs, and raw model response bodies must never be committed. See the
[model contract](docs/model-contract.md) for proxy fail-closed behavior and receipt rules.

> [!WARNING]
> Every configured-model generation or repair request may incur cost. `migrate` sends the complete input source;
> gap/operator paths send the bounded gap contract, statically extracted semantic function, applicable draft or
> diagnostics, and selected CANNBot Skill context; AsNumpy adapter paths send target-projected source/header
> context, Skill context, overlays, and applicable diagnostics. CANNBot supplies pinned workflow context—the
> configured remote model generates the draft. `verify-script`, `benchmark-script`, and `run_demo.sh` execute the
> original program and candidate with the current user's permissions. `python -I` is **not** an OS sandbox. Send
> only material you may disclose, run only trusted programs, and isolate untrusted input in a container or VM.

## Run the ordinary flow

The input must be trusted and assign its final result to a named variable (here, `result`):

```bash
SOURCE=/absolute/path/to/trusted_script.py
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/numpilot-run.XXXXXX")"

python -m numpilot analyze "$SOURCE" --require-direct --out "$RUN_DIR/analysis.json"
python -m numpilot migrate "$SOURCE" \
  --mode script --engine llm --backend npu --fallback-policy reject \
  --out "$RUN_DIR/migration"
python -m numpilot verify-script "$SOURCE" \
  --migration-report "$RUN_DIR/migration/script-migration-report.json" \
  --output result --out "$RUN_DIR/verification.json"
```

`./run_demo.sh --quick` is the repository's staged demonstration wrapper and additionally runs a scoped benchmark.
By default, each run uses a new repository-external state directory; inspect configured locations with
`python -m numpilot paths --json`. Benchmark claims require the separate
[strict-NPU benchmark contract](docs/script-benchmark.md).

The analyzer records an independent decision for every callsite, then exposes a fail-closed aggregate result for
the complete workload using the same five-state vocabulary:

| State | Meaning |
|---|---|
| `DIRECT` | The exact runtime profile has qualified the requested surface |
| `FALLBACK_CONSENT` | An audited CPU/host route exists and requires explicit user consent |
| `GAP_REQUIRED` | The API is missing, or the user rejected an audited fallback |
| `CAPABILITY_AUDIT_REQUIRED` | A callable exists, but its runtime identity or provenance is not qualified |
| `BLOCKED` | Dynamic or unmodeled Python behavior cannot be migrated safely |

## Runtime support and evidence scope

The package metadata supports Python 3.10+, but that is not the strict-NPU compatibility matrix. The current
strict profile is bound to Linux/AArch64, Ascend 910B4, CANN 9.0.0, CPython 3.10.20, and the pinned AsNumpy
source/install/receipt hashes. Any other hardware, CANN, Python ABI, wrapper, native library, or receipt remains
basic-install or candidate evidence until independently audited; no unpublished working log is release evidence.

Curated development evidence remains available for audit:

- [Poisson-Ritz staged evidence](evidence/competition-poisson-stage-20260828/README.md)
- [Elementwise advantage-range evidence](evidence/competition-ordinary-20260828/README.md)
- [Evidence index and offline audit commands](evidence/README.md)

These receipts are development evidence and must not be generalized to different hardware, CANN, Python,
AsNumpy, input shape, dtype, or timing scope.

## Branch and release model

Repository policy permits only two long-lived branches:

- `main`: the release branch; tags and release artifacts come only from a verified `main` commit.
- `develop`: the integration branch and normal contribution/validation target.

External work should use a fork and open a pull request into `develop`; release promotion is a reviewed
`develop → main` pull request. Both branches **must** be protected against force-push, history rewriting, and
unreviewed direct changes in GitCode. Local refs are aligned; server-side default-branch, head, and protection
settings still require an administrator check. See the
[branch governance and release policy](docs/SUBMISSION.md).

## Documentation

- [Installation and verification](docs/installation.md)
- [Current architecture and implementation boundaries](docs/architecture.md)
- [Documentation index](docs/README.md)
- [Numerical verification contract](docs/script-verification.md)
- [Strict-NPU benchmark contract](docs/script-benchmark.md)
- [Gap recovery state machine](docs/gap-recovery-state.md)
- [Changelog](CHANGELOG.md)
- [Issue tracker](https://gitcode.com/li-xingHIT/numpilot/issues)
- [Security policy](SECURITY.md)
- [Contribution guide](CONTRIBUTING.md)

## License and trademarks

NumPilot is licensed under [Apache License 2.0](LICENSE). Third-party components retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the
[brand asset inventory](docs/assets/brand/README.md). Product names and logos are used for technical
identification only, remain the property of their respective owners, and do not imply endorsement. Publication
of the screenshot-derived assets requires the repository owner to confirm the applicable trademark/use rights.
