# Install and verify NumPilot

[简体中文](installation.zh-CN.md) · [README](../README.md)

This guide separates three different outcomes. Do not treat one as proof of another.

| Outcome | What it proves | NPU required |
|---|---|---:|
| CPU/shim development | CLI, shim smoke, and environment-independent checks can run | No |
| Basic NPU installation | CANN, NPU, and the pinned AsNumpy build can execute a smoke test | Yes |
| Strict-NPU readiness | The complete runtime identity matches an audited profile for the requested calls | Yes, exact profile |

> [!WARNING]
> The current strict profile is bound to Linux/AArch64, Ascend 910B4, CANN 9.0.0, CPython 3.10.20, and exact
> AsNumpy source/install/receipt hashes. Python 3.10+ is accepted by the package and installer, but a different
> SoC, CANN, Python ABI, wrapper, native library, or receipt must return `CAPABILITY_AUDIT_REQUIRED` until it is
> independently qualified.

## 1. CPU/shim development setup

Use this path to inspect the project or contribute without an Ascend host:

```bash
git clone --branch main --depth 1 https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -m numpilot --version
python -m numpilot doctor --backend shim
env -u PYTHONPATH python -m pytest -q -p no:cacheprovider \
  tests/test_install_script.py tests/test_notebook_quickstart.py
```

The package metadata requires Python 3.10+ and installs `numpy>=1.26`, `loguru>=0.7`, and `rich>=13`.
`doctor --backend shim` is a development check and does not prove NPU capability.
The complete suite is a pinned release-matrix gate: isolate CANN-injected `PYTHONPATH` and proxy variables and
use the release Python environment. Do not treat an arbitrary Python 3.10+ host as a full-suite qualification.

## 2. NPU prerequisites

For the frozen-component mode, prepare these outside NumPilot before running `install.sh`:

- a compatible Ascend driver, firmware, CANN Toolkit, and OPP;
- a visible NPU (`npu-smi info` must work);
- Python 3.10+ with `venv`/`pip`;
- user-owned space for the virtual environment, persistent state, and runtime receipts.

Installing the fixed wheel does not require Git, CMake, Make/Ninja, GCC/G++, Python development headers, or
system CMake packages for `fmt` and `spdlog`. Those tools are needed only for the explicit maintainer
source-first build; on Ubuntu, the host administrator may install them with:

```bash
sudo apt-get update
sudo apt-get install -y \
  git cmake make ninja-build g++ python3-dev python3-venv libfmt-dev libspdlog-dev
```

Package names differ by distribution. CANN, the driver, firmware, and OPP must be installed according to the
target Ascend/CANN documentation; NumPilot does not install or modify them.

Check the fixed-component prerequisites before starting:

```bash
python3 --version
npu-smi info
test -f /usr/local/Ascend/cann-9.0.0/set_env.sh
```

For source-first mode, additionally check `git --version`, `g++ --version`, `cmake --version`, Python
`Python.h`, and the `fmt`/`spdlog` development packages.

Replace the CANN path with the exact installation you intend to audit. Choosing `latest` may be convenient for a
basic smoke test, but it does not make a newer CANN version part of the current strict profile.

## 3. Frozen internal AsNumpy component (release/competition mode)

Use this mode for a released NumPilot build. The repository already carries the companion under
`components/asnumpy/` (or it may be copied from the same immutable release bundle). Its manifest binds the
wheel, upstream LICENSE, SBOM, METADATA/RECORD, and installed native hashes. Do not use a PyPI `latest`,
a version range, or a moving Git branch. A missing or mismatched wheel is a release-blocking failure; the
installer must not fall back to downloading source.

```bash
python -m numpilot bootstrap --apply \
  --asnumpy-component components/asnumpy \
  --receipt /secure/runtime/bootstrap-receipt.json
python -m numpilot bootstrap --verify --json
```

The release manifest must also bind the wheel `METADATA`/`RECORD`, installed `asnumpy/__init__.py`,
native `_core` hashes, Python/architecture tag and the audited CANN/SoC profile. Keep the upstream
Apache-2.0 LICENSE beside the component and include it in the SBOM/license inventory. See
[AsNumpy internal-component lock](ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md) for the reference identity,
artifact caveat and upgrade/rollback procedure.

The current attachment targets Linux AArch64, CPython 3.10, and CANN 9.0.0. It does not include the CANN
driver, firmware, or OPP. You no longer need to fetch or compile AsNumpy on site, but the local wheel still
has to be installed once into the target virtual environment after loading the matching CANN environment.

The source-first flow below is retained only to build and audit a future component in an isolated
development workspace. It may fetch the **pinned** source when explicitly authorized; that behavior is
not the user-facing contract for a frozen release.

## 4. Source-first NPU installation (development/build preparation)

Use `main` for a release candidate and `develop` only for integration work. The current Alpha snapshot is not a
verified public package release, so do not substitute `pip install numpilot` for this flow.

Choose two absolute directories owned exclusively by the current user. Keeping both outside the repository makes
source review, cleanup, and runtime evidence boundaries clearer.

```bash
git clone --branch main --depth 1 https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot

./install.sh --allow-network \
  --asnumpy-source /absolute/path/to/pinned/asnumpy \
  --cann-env /usr/local/Ascend/cann-9.0.0/set_env.sh \
  --venv /absolute/user-owned/numpilot-venv \
  --persistent-root /absolute/user-owned/numpilot-runtime

. /absolute/user-owned/numpilot-runtime/activate-numpilot.sh
```

`--allow-network` is mandatory, explicit consent. The installer may use the selected Python package index and,
when a cache is missing, fetch the pinned AsNumpy source. It never runs `sudo`, edits shell startup files, stores
model credentials, or installs into system Python.

Useful alternatives:

```bash
# Use a clean checkout that already matches the exact pinned AsNumpy origin/commit/tree.
./install.sh --allow-network \
  --asnumpy-source /absolute/path/to/pinned/asnumpy \
  --cann-env /absolute/path/to/CANN/set_env.sh \
  --venv /absolute/user-owned/numpilot-venv \
  --persistent-root /absolute/user-owned/numpilot-runtime

# In frozen-component mode, select an explicit Python package index. Do not put credentials in the URL.
./install.sh --allow-network \
  --pip-index-url https://mirror.example.invalid/simple \
  --asnumpy-component "$PWD/components/asnumpy" \
  --cann-env /absolute/path/to/CANN/set_env.sh \
  --venv /absolute/user-owned/numpilot-venv \
  --persistent-root /absolute/user-owned/numpilot-runtime
```

The source build is not an OS-level network sandbox and the dependency ranges are not a complete release lock.
A pinned Git commit also does not guarantee byte-for-byte identical local wheels; a formal release must preserve
the exact wheel, hash, constraints, and wheelhouse used for qualification.

## 5. Post-install verification

Run every check from the generated activation environment.

### 5.1 Identity and isolated dependencies

```bash
python -m numpilot --version
python -m numpilot paths --json
env -u PYTHONPATH python -m pip --isolated check
```

Unsetting `PYTHONPATH` isolates the project virtual environment from optional Python packages injected by CANN.
A failure from CANN's vendor packages in the fully injected environment is not, by itself, proof that the
NumPilot virtual environment is broken.

### 5.2 Bootstrap receipt and AsNumpy

```bash
NUMPILOT_RECEIPT="$(python -m numpilot paths --get bootstrap_receipt)"
python -m json.tool "$NUMPILOT_RECEIPT"
python -m numpilot bootstrap --verify --json
```

The apply receipt should have `status="ready"` and `ready=true`, the pinned repository/commit/tree identity, and
a passed AsNumpy verification. `bootstrap --verify --json` should report the official `(1000, 1000)` `float32`
smoke result and non-empty native-module identities. Always review both the apply receipt and the live verify
result: standalone verify collects current hashes but does not reconstruct provenance on its own.

For an additional direct local check:

```bash
python - <<'PY'
from importlib.metadata import version
import asnumpy as ap

device = ap.ones((1000, 1000), dtype=ap.float32)
host = device.to_numpy()
assert host.shape == (1000, 1000)
assert str(host.dtype) == "float32"
print("asnumpy", version("asnumpy"), "smoke=PASS")
PY
```

This proves only that the local AsNumpy/NPU path can execute the smoke operation.

### 5.3 NPU doctor and strict-profile decision

```bash
python -m numpilot doctor --backend npu

python -m numpilot analyze examples/poisson_source_smooth_l1_map.py \
  --require-direct --json
```

`doctor` checks basic CANN/NPU/AsNumpy operation; use `doctor --json` when a machine-readable report is
needed. It is not a strict-profile qualification. `analyze --require-direct` is the next gate. An exit code of `1` with
`CAPABILITY_AUDIT_REQUIRED` is an honest runtime-profile drift result, not an installation error to bypass with
`--skip-env-check`.

## 6. Optional CANNBot Skills

The ordinary direct path does not need CANNBot. Install it only for the gap/operator path after reviewing the
official [CANNBot LICENSE](https://gitcode.com/cann/cannbot-skills/blob/master/LICENSE) and accepting **CANN Open
Software License Agreement Version 2.0**. For exact pinned terms, inspect
`git show 1ef6372a538f39b0b7ed34666a3e1f465fb4936a:LICENSE` in a clean CANNBot checkout before passing the
acceptance flag.

Install AsNumpy and the pinned CANNBot checkout together:

```bash
./install.sh --allow-network \
  --with-cannbot --accept-cannbot-license \
  --cann-env /absolute/path/to/CANN/set_env.sh \
  --venv /absolute/user-owned/numpilot-venv \
  --persistent-root /absolute/user-owned/numpilot-runtime
```

Or add the pinned checkout after the base installation:

```bash
python -m numpilot setup-skills --accept-cannbot-license
python -m numpilot setup-skills --check
python -m numpilot bootstrap --verify \
  --with-cannbot --accept-cannbot-license
```

The pinned identity is `cann/cannbot-skills@1ef6372a538f39b0b7ed34666a3e1f465fb4936a`. Verification checks the
exact origin, HEAD, and the bounded set of Skill files that NumPilot consumes. It does not certify every file or
every Skill in that repository.

To verify an existing exact checkout without fetching:

```bash
python -m numpilot setup-skills \
  --root /absolute/path/to/cannbot-skills --check
```

CANNBot Skills and `cann/ops-nn` are different dependencies. The installer does not automatically fetch
`ops-nn`; the current official-source resolver accepts only an explicitly provided, version-matched local
checkout and currently pins the CANN 9.0.0 source mapping.

## 7. Configure and probe the model

```bash
python -m numpilot configure-llm --init
# Edit the private schema-2 llm.json path printed by the command.
python -m numpilot configure-llm --check
python -m numpilot configure-llm --probe
```

`--check` is network-free and validates both local configuration and the current proxy gate. `--probe` performs
a real request, may incur cost, and requires the expected readiness marker. Configuration needs `base_url`,
`auth_token`, and `model`; `provider` is only an audit label, and the current protocol is
`openai-chat-completions`.

The configuration directory must be `0700` and `llm.json` must be `0600`. Never commit the token, endpoint,
proxy URL, or raw response body.

> [!WARNING]
> Every configured-model generation or repair call may incur cost. `migrate` sends the complete input source.
> Gap/operator generation and repair send a bounded gap contract, the statically extracted semantic function,
> selected pinned CANNBot Skill content, and applicable current files or diagnostics. AsNumpy adapter generation
> and repair send only the source/header context selected by the local target projection, the repository-owned
> integration Skill, overlays, and applicable diagnostics. CANNBot provides workflow context; the configured
> remote model—not CANNBot itself—generates the unverified draft. Review these payload classes before using a
> non-local endpoint.
>
> `verify-script`, `benchmark-script`, and the demo execute the original source and generated candidate with your
> user permissions. `python -I` process isolation is not an OS sandbox. Send only material you may disclose and
> run only programs you trust; place untrusted inputs in a dedicated container or VM.

If the endpoint must use a proxy:

```bash
export HTTP_PROXY=http://127.0.0.1:PORT
export HTTPS_PROXY=http://127.0.0.1:PORT
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NUMPILOT_LLM_REQUIRE_PROXY=1
python -m numpilot configure-llm --probe
```

The fail-closed proxy mode rejects a missing protocol-matched proxy or a matching `NO_PROXY` bypass before
network transport begins.

## 8. First run

Use the complete `analyze → migrate → verify-script` example in the [English README](../README.md#run-the-ordinary-flow)
for your own trusted script. `./run_demo.sh --quick` runs the repository's staged example and adds a scoped
benchmark. If the model config lives in a separate runtime directory, pass
`--llm-config /absolute/persistent/numpilot-runtime/.numpilot-config/llm.json`; each run is written to a new
repository-external directory shown by `python -m numpilot paths --json`.

## 9. Troubleshooting

| Symptom | Check |
|---|---|
| `Python.h` missing | Install the Python development headers for the selected interpreter |
| `venv` creation fails | Install the distribution's Python venv package |
| `fmt` or `spdlog` CMake package missing | Install the system development package; NumPilot will not run `sudo` |
| CANN environment missing | Pass the exact `--cann-env /.../set_env.sh`; avoid mixing multiple CANN paths |
| `npu-smi` or `libascendcl.so` missing | Fix the driver/CANN installation outside NumPilot |
| `bootstrap --verify` fails | Review the apply receipt, module paths, native hashes, and generated activation file |
| `doctor` passes but analysis requires audit | Basic install succeeded, but the strict profile does not match |
| CANNBot check fails | Verify exact origin, commit, clean tracked Skill files, cache path, and license decision |
| Model probe fails | Check private config, TLS, exact response model, proxy variables, and `NO_PROXY` |
| Custom receipt path rejected | Its parent must already exist, be user-owned, non-symlink, and not group/world writable |

Source the generated activation file in a fresh shell rather than manually combining multiple CANN versions.
Sourcing it twice is supported and removes known toolkit/devlib conflicts.

## 10. Safe cleanup

NumPilot has no `uninstall` command. First record the exact paths:

```bash
python -m numpilot paths --json
```

While the dedicated NumPilot virtual environment is still active, optionally remove only its packages with:

```bash
python -m pip uninstall numpilot asnumpy
deactivate
```

Then review the dedicated virtual-environment and persistent-root absolute paths before removing either path.
Package removal does not delete the private LLM token file reported by
`python -m numpilot paths --get llm_config`; remove that exact file separately if desired.

The installer never modifies CANN, the driver, firmware, OPP, or shell startup files, so those components are
outside NumPilot cleanup. Do not run a broad recursive deletion against a home directory or an unverified path.
