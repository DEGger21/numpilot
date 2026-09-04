<div align="center">

<h1><img src="docs/assets/brand/numpilot-wordmark.png" alt="NumPilot" width="560"></h1>

### Verified NumPy migration for Huawei Ascend NPUs

<p>
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="docs/installation.md">Installation</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="CONTRIBUTING.md">Contributing</a> ·
  <a href="LICENSE">License</a>
</p>

<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.3.0a1.dev0-6D5CE7?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-F59E0B?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white">
  <img alt="NPU platform" src="https://img.shields.io/badge/NPU-Linux%20AArch64-008AA3?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-2EA44F?style=flat-square">
</p>

</div>

NumPilot analyzes supported, statically understandable single-file NumPy programs, produces AsNumpy-based
migration candidates, and admits them only after deterministic capability, provenance, structure, and numerical
checks. Unsupported or ambiguous behavior stops explicitly instead of being silently rewritten or routed to CPU.

> The model proposes a candidate; NumPilot's deterministic gates decide whether it can be used.

## Install

### Core CLI

This route supports analysis, the shim backend, and Agent Skill use without CANN or an NPU:

```bash
git clone --branch main --depth 1 https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m numpilot doctor --backend shim
```

### Ascend NPU runtime

Install the matching Ascend driver and CANN first. The bundled component requires Linux/AArch64 and CPython
3.10; the current strict profile was verified with CPython 3.10.20 and CANN 9.0.0 at
`/usr/local/Ascend/cann-9.0.0`. Check the selected interpreter before installing:

```bash
uname -m
python3 -c 'import platform; print(platform.python_implementation(), platform.python_version())'
mkdir -p "$HOME/.local/venvs"
chmod 700 "$HOME/.local/venvs"

./install.sh --allow-network \
  --python "$(command -v python3)" \
  --cann-env /usr/local/Ascend/cann-9.0.0/set_env.sh \
  --venv "$HOME/.local/venvs/numpilot" \
  --persistent-root "$HOME/.local/share/numpilot"

. "$HOME/.local/share/numpilot/activate-numpilot.sh"
python -m numpilot bootstrap --verify --json
python -m numpilot doctor --backend npu
```

The two checks must report `aarch64` and `CPython 3.10.x`. Use
`--python /absolute/path/to/python3.10` when the correct interpreter is not the default `python3`; only the exact
3.10.20 profile currently carries the strict verified claim.

What these commands add:

| Route | Installed or created | Not installed |
|---|---|---|
| `pip install .` | NumPilot CLI, NumPy, Loguru, Rich, the Skill bundles, and the locked AsNumpy component attachment as package data | Ascend driver, CANN, OPP, an active AsNumpy runtime, CANNBot, or an LLM |
| `install.sh` | A private virtual environment, NumPilot and its Python dependencies, the exact bundled AsNumpy wheel, bootstrap receipt, activation file, and state/config/cache directories | Ascend driver, firmware, CANN, OPP, system native libraries, model credentials, or shell-startup changes |

CANNBot is optional. Enable the pinned checkout with `--with-cannbot --accept-cannbot-license`, or select an
existing exact checkout with `--cannbot-source /absolute/path/to/cannbot-skills --accept-cannbot-license`.
The configured-terminal `add_one` gap example requires one of these forms; ordinary mappings that already have
qualified operators do not require CANNBot. The installer uses an editable NumPilot install and binds the
activation file to this checkout, so keep the checkout at the same path. See the
[installation guide](docs/installation.md) for prerequisites and troubleshooting.

## Choose a workflow

| Workflow | Model source | Is `llm.json` required? |
|---|---|---|
| Terminal workflow | The OpenAI-compatible endpoint configured for NumPilot | Yes |
| Agent Skill workflow | The AI already running in your IDE or sidebar | No; the agent must not read or invoke it |

### A. Terminal workflow with `llm.json`

Ask NumPilot for the exact configuration path, then create the private template:

```bash
python -m numpilot paths --get llm_config
python -m numpilot configure-llm --init
```

Without path overrides the file is `~/.config/numpilot/llm.json` (or
`$XDG_CONFIG_HOME/numpilot/llm.json`). After activating the NPU installation shown above it is
`$HOME/.local/share/numpilot/.numpilot-config/llm.json`. The path printed by `paths --get llm_config` is
authoritative. Fill it as follows:

```json
{
  "schema_version": 2,
  "base_url": "https://your-openai-compatible-service.example/v1",
  "auth_token": "REPLACE_WITH_A_PRIVATE_TOKEN",
  "model": "exact-model-id",
  "provider": "provider-label",
  "protocol": "openai-chat-completions",
  "response_model_aliases": []
}
```

Then validate locally and make one explicit network probe:

```bash
python -m numpilot configure-llm --check
python -m numpilot configure-llm --probe
numpilot
```

Inside the shell, enter `check llm`, `migrate`, the absolute source path, `y`, `execute`, and `exit` in that
order. `y` authorizes generation; `execute` separately authorizes running the displayed hash-bound source and
candidate. Keep the configuration directory at mode `0700` and `llm.json` at `0600`; never commit it.

### B. Agent Skill workflow without `llm.json`

Open the cloned repository in an IDE/sidebar AI that can read workspace files and run approved commands. Do
not run `configure-llm` or `check llm`. Send the sidebar AI a prompt like this:

```text
Read /absolute/path/to/numpilot/skill/SKILL.md and use its agent/no-llm lane.
Migrate /absolute/path/to/source.py with the model already available in this sidebar.
Do not read llm.json, do not run configure-llm or check llm, and do not invoke NumPilot's
configured-model provider. First perform read-only analysis and show me the plan. After I confirm,
author the candidate yourself and submit only supported artifacts to NumPilot's deterministic gates.
Report any stopped, unverified, numeric, runtime, and performance states without upgrading them.
```

For the bundled example, replace the source with the absolute path to
`examples/add_one_migration_demo.py`. The Skill is an operating contract, not a model: the sidebar AI writes
the candidate, while NumPilot performs the supported analysis and verification gates. Without an NPU runtime,
this route can analyze and draft but cannot claim strict-NPU qualification. A route without a registered
artifact adapter stops honestly at an agent-authored draft rather than fabricating later receipts.

<p align="center">
  <img src="docs/assets/architecture/numpilot-two-routes.zh-CN.png" alt="Terminal and Agent Skill workflows" width="100%">
</p>

## Highlights

- Whole-file call-site and bounded data-flow analysis for the supported NumPy subset.
- Runtime-bound capability matching across AsNumpy, CANN, and the detected Ascend SoC.
- Model-assisted migration with response, AST, source, and artifact integrity checks.
- Official-operator-first discovery, registered gap-resolution workflows, and constrained operator drafting.
- Isolated numerical comparison and real-NPU qualification for supported profiles.
- Scope-bound benchmarking and machine-readable JSON/Markdown reports.
- Interactive terminal and Agent Skill workflows.

## Architecture overview

<p align="center">
  <a href="docs/numpilot-zh-v6.svg">
    <img src="docs/numpilot-zh-v6.svg" alt="NumPilot v6 architecture" width="100%">
  </a>
</p>

<p align="center"><sub>Click the diagram to open the full-size architecture view.</sub></p>

## How it works

```text
Analyze source → Match runtime capabilities → Generate candidate → Qualify → Verify → Report
```

Every stage is fail-closed. A generated operator remains an unverified draft until its required build, test,
public-API, numerical, and real-device gates have passed.

## Compatibility

NumPilot's installer accepts user-owned paths for CANN, virtual environments, state, cache, and model settings;
it does not depend on one developer's workspace. The current strict-NPU profile nevertheless binds the exact
CANN identity and location shown below. Binary NPU compatibility is intentionally narrower than source-level
CLI compatibility:

| Mode | Requirements | Supported use |
|---|---|---|
| Core CLI / shim | Python 3.10+, NumPy 1.26+, Linux | Analysis, CLI exploration, and numerical diagnostics without an NPU; no strict-NPU claim |
| Bundled NPU component | Linux/AArch64, CPython 3.10, CANN 9.0.0, compatible Ascend driver/firmware/OPP | Installs the hash-bound AsNumpy wheel shipped in `components/asnumpy/` |
| Strict verified profile | Ascend 910B4, CANN 9.0.0 at `/usr/local/Ascend/cann-9.0.0`, CPython 3.10.20, NumPy 2.2.6, Loguru 0.7.3, bundled AsNumpy component | Real-NPU qualification and scope-bound performance gates |
| Other machines | Different SoC, CANN version/location, Python ABI, operating system, or native component | Not automatically qualified; define and review a new runtime profile, then rebuild and validate its component before making strict-NPU claims |

The bundled wheel is tagged `linux_aarch64`, not `manylinux`, and has not been validated across Linux
distributions. Its current native build requires glibc 2.34+, a compatible libstdc++, `libfmt.so.8`,
`libspdlog.so.1`, `libpython3.10.so.1.0`, and the matching CANN runtime libraries.
The package metadata's Python 3.10+ declaration alone therefore does not establish NPU compatibility. On every
new machine, run `bootstrap --verify` and `doctor`; these checks validate the installation, while each workload
must still pass its own capability and qualification gates. NumPilot does not install the Ascend driver,
firmware, CANN, or OPP.

## Security notes

- Migrate and execute only source code you trust. `python -I` isolation is not an operating-system sandbox.
- A configured remote model may receive source code and may incur usage charges.
- Generated kernels and integrations remain unverified until all required gates pass.
- Runtime identity or artifact drift produces an explicit blocked/audit-required result.

Report vulnerabilities through [SECURITY.md](SECURITY.md).

## Documentation

- [Installation and verification](docs/installation.md)
- [Architecture and trust boundaries](docs/architecture.md)
- [Model configuration and transport contract](docs/model-contract.md)
- [Numerical verification](docs/script-verification.md)
- [Strict-NPU benchmarking](docs/script-benchmark.md)
- [Agent Skill contract](skill/SKILL.md)
- [Sidebar workflow](skill/references/codearts-sidebar.md)
- [Changelog](CHANGELOG.md)

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), add tests for behavioral changes, and run
the relevant test and release-tree checks before opening a pull request.

## License

NumPilot is licensed under the [Apache License 2.0](LICENSE). Third-party components retain their own licenses;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Product names and logos remain the property of their
respective owners and do not imply endorsement. Confirm the redistribution terms for bundled binaries and brand
assets before publishing a binary distribution.
