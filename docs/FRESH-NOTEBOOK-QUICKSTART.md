# 全新 Notebook 验收路径（Demo v1）

这份流程用于在一台已经安装 CANN 9.0.0、可访问 Ascend NPU 的机器上，从全新目录验证
NumPilot `v0.2.0a1`。对应的无输出 Notebook 模板是
[NumPilot-Demo-v1-Quickstart.ipynb](NumPilot-Demo-v1-Quickstart.ipynb)。

本次验收不接入 ops-nn，也不现场构建 AsNumpy。它固定走经过审计的 CPython 3.10/aarch64
P0 wheel，依次验证：

```text
clone tag
  → 安装 NumPilot[demo]
  → bootstrap plan/apply
  → AsNumpy RECORD + 安装后 module/native identity
  → AsNumpy README smoke + CANNBot 本地 Skill 清单
  → GLM-5.3 普通 NumPy 脚本迁移
  → exact migrated.py 在 NPU 上执行
```

## 0. 取得 Notebook 与发布物

Notebook 模板位于发布 tag 的
`docs/NumPilot-Demo-v1-Quickstart.ipynb`。可以直接使用 Demo v1 发布附件里的模板；如果先
clone 仓库只是为了取得模板，后续必须把 `NUMPILOT_NOTEBOOK_ROOT` 设为该 clone 之外、尚不
存在的绝对目录。Notebook 会在这个 write-once 目录中再次执行固定 tag clone，形成独立验收
环境，不会把 venv、cache 或迁移输出写入模板仓库。

同机交付包/发布附件名称固定为
`numpilot-demo-v1-cp310-aarch64-wheelhouse.tar.gz`，校验文件为 `SHA256SUMS`。解压后的
`wheelhouse/` 同时包含 NumPilot wheel、P0 AsNumpy wheel 和全部 Python 运行依赖。启动
Notebook 前建议设置：

```bash
export NUMPILOT_NOTEBOOK_ROOT=/absolute/fresh/path/numpilot-demo-v1-run
export NUMPILOT_WHEELHOUSE=/absolute/path/to/wheelhouse
export NUMPILOT_ASNUMPY_WHEEL="$NUMPILOT_WHEELHOUSE/asnumpy-0.1.dev106+g21368abf9.d20260822-cp310-cp310-linux_aarch64.whl"
export NUMPILOT_CONFIG_HOME=/absolute/path/to/directory-containing-llm-json
export NUMPILOT_HTTP_PROXY=<proxy-url>
```

先在交付目录执行 `sha256sum -c SHA256SUMS`。不要把代理、Token 或 `llm.json` 复制到仓库、
wheelhouse 或 Notebook 中。

## 1. 前置条件

- CPython 3.10、`python3.10 -m venv`/ensurepip、CANN 9.0.0、Ascend 910B 和 `npu-smi`
  已可用；
- NumPilot Demo v1 wheelhouse 或可访问 Python 包索引；
- P0 AsNumpy wheel，SHA-256 必须是
  `afe845c45d17e7f5cd619831f555fba3c2344296ffe02f1d251017b04cf0282d`；
- 已审阅固定 CANNBot commit `abcb35fc0ff77f4bc5eda74de6bbcf28cbc1572b` 的
  [CANN Open Software License Agreement v2.0](https://gitcode.com/cann/cannbot-skills/blob/abcb35fc0ff77f4bc5eda74de6bbcf28cbc1572b/LICENSE)，
  并愿意为 CANNBot 获取和复验显式传入 `--accept-cannbot-license`；
- 私有 `llm.json` 已配置 `provider=glm`、`model=glm-5.3`，权限为 `0600`；
- GLM/GitCode 访问所需代理只注入当前 Notebook 进程，不写入仓库或配置文件。

如果没有 P0 wheel 或运行环境不是 CPython 3.10/aarch64，请停止这条验收路径。不要用
`--build-asnumpy` 冒充同一 Demo profile，也不要沿用 Case 01 的 NPU 性能结论。

## 2. Notebook 第一个单元：参数与隔离环境

```python
from pathlib import Path
import math
import json
import os
import re
import shlex
import subprocess

root_text = os.environ.get("NUMPILOT_NOTEBOOK_ROOT") or input(
    "Fresh absolute test root outside the template clone: "
).strip()
root_path = Path(root_text).expanduser()
assert root_text and root_path.is_absolute(), "NUMPILOT_NOTEBOOK_ROOT must be absolute"
TEST_ROOT = root_path.resolve()
REPO = TEST_ROOT / "numpilot"
VENV = TEST_ROOT / ".venv"
PYTHON = VENV / "bin" / "python"

# 优先通过 Notebook 启动环境提供；否则在运行本单元时输入，不会写回文件。
proxy = os.environ.get("NUMPILOT_HTTP_PROXY") or input("HTTP(S) proxy: ").strip()
wheel_text = os.environ.get("NUMPILOT_ASNUMPY_WHEEL") or input(
    "P0 AsNumpy wheel absolute path: "
).strip()
config_text = os.environ.get("NUMPILOT_CONFIG_HOME") or input(
    "Directory containing private llm.json: "
).strip()
ASNUMPY_WHEEL = Path(wheel_text).expanduser().resolve()
LLM_CONFIG_HOME = Path(config_text).expanduser().resolve()
ASNUMPY_SHA256 = "afe845c45d17e7f5cd619831f555fba3c2344296ffe02f1d251017b04cf0282d"

CANN_HOME = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0"))
WHEELHOUSE_TEXT = os.environ.get("NUMPILOT_WHEELHOUSE", "").strip()
WHEELHOUSE = Path(WHEELHOUSE_TEXT).expanduser().resolve() if WHEELHOUSE_TEXT else None

assert not TEST_ROOT.exists(), f"write-once test root already exists: {TEST_ROOT}"
for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
    if (candidate / ".git").exists():
        try:
            TEST_ROOT.relative_to(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError(f"test root must be outside template repository: {candidate}")
        break
assert ASNUMPY_WHEEL.is_file(), ASNUMPY_WHEEL
assert (LLM_CONFIG_HOME / "llm.json").is_file(), LLM_CONFIG_HOME / "llm.json"
assert CANN_HOME.is_dir(), CANN_HOME

env = {
    name: value
    for name, value in os.environ.items()
    if not name.startswith("NUMPILOT_")
}
for name in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
):
    env.pop(name, None)
env.update({
    "http_proxy": proxy,
    "https_proxy": proxy,
    "ASCEND_HOME_PATH": str(CANN_HOME),
    "ASCEND_TOOLKIT_HOME": str(CANN_HOME),
    "ASCEND_OPP_PATH": str(CANN_HOME / "opp"),
    "ASCEND_AICPU_PATH": str(CANN_HOME),
    "NUMPILOT_CONFIG_HOME": str(LLM_CONFIG_HOME),
    "NUMPILOT_STATE_HOME": str(TEST_ROOT / "state"),
    "NUMPILOT_LLM_MAX_TOKENS": "8192",
    "NUMPILOT_LLM_TIMEOUT": "300",
    "NUMPILOT_LLM_INSECURE": "0",
    "XDG_CACHE_HOME": str(TEST_ROOT / "cache"),
    "PYTHONNOUSERSITE": "1",
})
env["PYTHONPATH"] = os.pathsep.join([
    str(CANN_HOME / "python/site-packages"),
    str(CANN_HOME / "opp/built-in/op_impl/ai_core/tbe"),
])
env["LD_LIBRARY_PATH"] = ":".join([
    "/usr/local/Ascend/driver/lib64",
    "/usr/local/Ascend/driver/lib64/common",
    "/usr/local/Ascend/driver/lib64/driver",
    str(CANN_HOME / "aarch64-linux/lib64"),
    str(CANN_HOME / "lib64/plugin/opskernel"),
    str(CANN_HOME / "lib64/plugin/nnengine"),
    str(CANN_HOME / "opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"),
])

def run(*args, cwd=None, capture=False):
    command = [str(item) for item in args]
    print("$", shlex.join(command))
    result = subprocess.run(
        command,
        cwd=cwd or TEST_ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if capture:
        print(result.stdout, end="")
    return result
```

如果机器的 CANN/driver 路径不同，只修改本单元。出现 `libmetadef.so undefined symbol` 通常表示
Notebook 继承了冲突的 `LD_LIBRARY_PATH`；不要继续，先修正上述 library 顺序并从新测试目录
重启。

## 3. Clone 固定 tag

```python
TEST_ROOT.mkdir(parents=True)
run(
    "git", "clone", "--branch", "v0.2.0a1", "--depth", "1",
    "https://gitcode.com/li-xingHIT/numpilot.git", REPO,
)
run("git", "-C", REPO, "status", "--short", "--branch")
run("git", "-C", REPO, "describe", "--tags", "--exact-match")
```

预期 tag 为 `v0.2.0a1`，工作树为空。不要改成 `master` 或未固定 commit。

## 4. 创建 Python 3.10 环境并安装

```python
run("python3.10", "-m", "venv", VENV)

if WHEELHOUSE is not None:
    assert WHEELHOUSE.is_dir(), WHEELHOUSE
    # -I prevents an old source checkout or CANN Python package metadata from
    # being mistaken for the package installed in this fresh venv.
    run(
        PYTHON, "-I", "-m", "pip", "--isolated", "install", "--no-index",
        "--find-links", WHEELHOUSE, "numpilot[demo]==0.2.0a1",
    )
else:
    # 这一步会通过显式 pip 命令访问包索引；不是 import/migrate 隐式下载。
    run(
        PYTHON, "-I", "-m", "pip", "--isolated", "install", "-e",
        f"{REPO}[demo]",
    )

run(PYTHON, "-m", "numpilot", "--version")
run(PYTHON, "-m", "numpilot", "paths", "--json")
```

推荐使用发布 wheelhouse；它应包含 NumPilot 及其 Python 依赖。AsNumpy 仍由下一步显式
bootstrap 安装，以保留 wheel/hash/receipt 门。

## 5. Bootstrap plan、apply 与真实复验

```python
RECEIPT = TEST_ROOT / "bootstrap-receipt.json"

run(
    PYTHON, "-m", "numpilot", "bootstrap", "--plan", "--json",
    "--asnumpy-wheel", ASNUMPY_WHEEL,
    "--asnumpy-wheel-sha256", ASNUMPY_SHA256,
    "--with-cannbot", "--accept-cannbot-license",
)

run(
    PYTHON, "-m", "numpilot", "bootstrap", "--apply",
    "--asnumpy-wheel", ASNUMPY_WHEEL,
    "--asnumpy-wheel-sha256", ASNUMPY_SHA256,
    "--with-cannbot", "--accept-cannbot-license",
    "--receipt", RECEIPT,
)

run(
    PYTHON, "-m", "numpilot", "bootstrap", "--verify",
    "--with-cannbot", "--accept-cannbot-license",
)
run(PYTHON, "-m", "numpilot", "doctor", "--backend", "npu")

receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
assert receipt["status"] == "ready" and receipt["ready"] is True
assert receipt["wheel"]["sha256"] == ASNUMPY_SHA256
assert receipt["verification"]["asnumpy"]["wheel_file_identity_verified"] is True
print(receipt["wheel"]["trust"]["profile_id"])
```

`bootstrap --verify` 应显示：

- `ap.ones((1000, 1000), dtype=float32)` 通过；
- AsNumpy/NumPy/Loguru 版本与 module/native hash；
- “现在写 Ascend C 算子有哪些能参考的 Skill？”及 8 个本地已校验 Skill；
- `network_requested_by_numpilot=false`（verify 本身不联网）。

## 6. 检查 GLM 配置并迁移普通 NumPy 文件

本次使用仓库内已经由 GLM 生成的 Case 01 大规模 NumPy 源文件作为输入；NumPilot 仍会通过
配置的 `glm-5.3` 生成一份全新的 write-once 迁移产物。不要复用仓库内精选迁移文件冒充本次
模型输出。

```python
llm_status = run(
    PYTHON, "-m", "numpilot", "configure-llm", "--check", capture=True
).stdout
assert "source=user-config" in llm_status
assert "provider=glm" in llm_status and "model=glm-5.3" in llm_status

SOURCE = REPO / "evidence/demotest-case01-20260825/source.py"
OUT = TEST_ROOT / "case01-migration"
run(
    PYTHON, "-m", "numpilot", "migrate", SOURCE,
    "--mode", "script", "--backend", "npu",
    "--fallback-policy", "reject",
    "--target-soc", "ascend910b", "--cann-version", "9.0.0",
    "--out", OUT,
)

report = json.loads((OUT / "script-migration-report.json").read_text(encoding="utf-8"))
assert report["status"] == "migrated_scoped_npu"
assert report["provider"] == "glm"
assert report["requested_model"] == report["response_model"] == "glm-5.3"
assert report["execution_performed"] is False
print(report["migrated_sha256"])
```

代理只由本 Notebook 的 `env` 传给 Git/GLM 子进程；Token 不出现在命令行或输出中。

## 7. CPU 与 NPU 实际运行

```python
cpu_output = run(PYTHON, SOURCE, capture=True).stdout
npu_output = run(PYTHON, OUT / "migrated.py", capture=True).stdout

pattern = re.compile(r"row_sum_checksum\s*=\s*\[\[\s*([-+0-9.eE]+)\s*\]\]")
cpu_match, npu_match = pattern.search(cpu_output), pattern.search(npu_output)
assert cpu_match and npu_match, "missing row_sum_checksum"
cpu_checksum, npu_checksum = float(cpu_match.group(1)), float(npu_match.group(1))
assert math.isclose(cpu_checksum, npu_checksum, rel_tol=1e-5, abs_tol=1e-4)
print(f"PASS: CPU/NPU checksum={npu_checksum}")
```

两侧应打印一致的 `row_sum_checksum`，NPU 侧会初始化 CANN 并执行 AsNumpy。这里是功能 smoke，
不是公平性能 benchmark；第一次 NPU 冷启动通常更慢。稳态 `180.004×` 结论只能引用仓库内
哈希绑定的 [Case 01 精选证据](../evidence/demotest-case01-20260825/README.md)，不能用这两个
冷进程命令重新推导。

## 8. 验收成功标志

以下条件必须同时成立：

1. clone 精确落在 `v0.2.0a1`；
2. NumPilot 版本为 `0.2.0a1`；
3. bootstrap receipt 为 `ready`，P0 profile、精确 Python/NumPy/Loguru、wheel RECORD 与
   安装后 module/native identity 均绑定；
4. AsNumpy README smoke 和 CANNBot 8 项本地 Skill 清单通过；
5. `doctor --backend npu` 无错误；
6. 新迁移报告绑定 `glm-5.3` 的请求/响应与产物 hash；
7. `migrated.py` 真正在 NPU 环境运行并输出正确 checksum。

任一步失败都保留 `TEST_ROOT` 和 receipt/report，不覆盖重跑。修复环境后换一个全新测试根目录；
不要人工改写 GLM 迁移产物，也不要在这条 Demo v1 验收里接入 ops-nn。
