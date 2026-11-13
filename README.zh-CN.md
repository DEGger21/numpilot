<div align="center">

<h1><img src="docs/assets/brand/numpilot-wordmark.png" alt="NumPilot" width="560"></h1>

### 面向昇腾 NPU 的端到端 NumPy → AsNumpy 迁移框架

<p>
  <a href="README.md">English</a> ·
  <a href="#快速安装与使用">快速安装</a> ·
  <a href="#系统架构">系统架构</a> ·
  <a href="#实测迁移与加速效果">实测效果</a> ·
  <a href="#贡献与许可证">贡献</a>
</p>

<p>
  <img alt="版本" src="https://img.shields.io/badge/version-0.3.0a1.dev0-6D5CE7?style=flat-square">
  <img alt="发布阶段" src="https://img.shields.io/badge/status-alpha-F59E0B?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/Python-as%20supported%20by%20AsNumpy-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white">
  <img alt="NPU" src="https://img.shields.io/badge/NPU-Ascend%20(runtime--detected)-008AA3?style=flat-square">
  <img alt="许可证" src="https://img.shields.io/badge/license-Apache--2.0-2EA44F?style=flat-square">
</p>

</div>

## 项目简介

NumPilot 是一个把完整 NumPy 程序端到端迁移为昇腾 NPU 上可执行的完整 AsNumpy 程序的迁移
框架，每个阶段都会产出可审计的回执。

[AsNumpy](https://github.com/asnumpy/asnumpy) 是面向昇腾 NPU 的 NumPy 兼容张量库，由
哈尔滨工业大学 AISS 团队（苏统华教授团队）开发。

## 快速安装与使用

NumPilot 提供两条安装路径。要复现比赛展示样例，请在昇腾服务器上使用静态组件路径；没有 CANN 的开发机使用 CPU/shim 路径。

**昇腾静态组件路径要求：** Linux AArch64、CPython 3.12、昇腾设备、CANN 9.0 或更高版本，以及当前用户拥有的虚拟环境目录。下方记录的 demo 结果使用经过审计的 CANN 9.2.0-beta.2 profile。

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

安装脚本会安装 components/asnumpy 中随仓库提供的精确 wheel，写入带哈希绑定的 bootstrap receipt，并检查 CANN/NPU 运行时；不会从网络下载浮动版本的 AsNumpy。不要在不同机器之间复制虚拟环境。

**CPU/shim 开发路径：**

~~~bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m numpilot doctor --backend shim
python -m pytest -q -p no:cacheprovider tests/test_install_script.py tests/test_bootstrap.py
~~~

迁移并验证完整程序：

~~~bash
numpilot analyze program.py
numpilot migrate program.py --out runs/demo
numpilot verify program.py \
  --migration-report runs/demo/migration/script-migration-report.json \
  --output result --out runs/demo/verification.json
~~~

如需使用配置模型补全缺口算子，复制模板并填入自己的服务信息。只有配置模型路径会发起模型请求；普通分析、验证、benchmark 和报告阶段使用确定性门禁与回执：

~~~bash
install -m 600 llm.example.json llm.json
# 编辑 llm.json：base_url、auth_token、model
python -m numpilot configure-llm --check
~~~

完整安装流程见[安装指南](docs/installation.zh-CN.md)。

### 交互式工作台

激活 NumPilot 环境后，从仓库根目录启动：

~~~bash
numpilot
~~~

启动页包含 NumPilot logo、模型状态栏和五个选项：新建迁移、加载迁移记忆、
检查模型、环境自检、退出。模型状态初始为 `未检查`；输入 `/modelcheck`（或选择
选项 3）会向 `llm.json` 配置的模型真实发送一次轻量探测，随后显示 `模型可用`
或 `模型不可用`。模型检查始终由用户显式触发，启动页本身不会调用模型。

| 命令 | 作用 |
| --- | --- |
| `/help` | 查看交互说明、状态规范和产物布局 |
| `/new` | 输入 NumPy 程序绝对路径并创建全新会话 |
| `/resume` | 加载历史迁移记忆 |
| `/readreport` | 在终端渲染最近一次 `migration-report.md` |
| `/modelcheck` | 手动检查配置模型 |
| `/doctor` | 检查 CANN、NPU、Python 与 AsNumpy |
| `/status` | 查看当前模型状态 |
| `/migrate /abs/x.py` | 已知绝对路径时直接开始迁移 |
| `/exit` | 退出 NumPilot |

每次迁移都会创建 `runs/demo-<时间戳>/`，并落盘本次会话的全部内容：
`source/<程序名>.py`（用户源码副本）、`migration-plan.md/json`（Analyzer/Planner
事实与模型迁移计划）、`migration-report.md/json`，以及 `workflow/` 回执与日志。
没有 NumPy 导入或调用的程序会在任何模型请求之前被拒绝。智能体状态是单调的：
已完成的角色不会跳回执行中；Diagnostics 和 Reporter 在模型诊断/报告步骤完成后
才显示完成。

### 运行 demo

激活 NumPilot 环境后，在仓库根目录执行。先配置 `llm.json`：

~~~bash
install -m 600 llm.example.json llm.json
${EDITOR:-vi} llm.json   # 填写 base_url、auth_token、model
python -m numpilot configure-llm --check
~~~

然后启动工作台：

~~~bash
numpilot
~~~

在交互界面中依次输入：

~~~text
/modelcheck
/new
/path/to/numpilot/examples/demo/large_numpy_program.py
y
/readreport
/resume
/exit
~~~

这里要填写仓库中 `examples/demo/large_numpy_program.py` 的绝对路径。
`/new` 会创建全新的 `runs/demo-<时间戳>/`；`y` 确认开始迁移；`/readreport`
在终端渲染最终 Markdown 报告；`/resume` 查看保存的迁移记忆。

如需非交互自动化，也可以直接执行同一条 demo 流程：

~~~bash
python -m numpilot interactive-migrate "$PWD/examples/demo/large_numpy_program.py" \
  --backend npu --benchmark --out "runs/demo-$(date +%Y%m%d-%H%M%S)" --yes --execute-source
~~~

`llm.example.json` 必须保持空白；`llm.json` 是被 git 忽略的私有文件。每位使用者
都要创建并配置自己的副本，不要复用其他人的 token 或 endpoint。

## 系统架构

<p align="center">
  <img src="docs/assets/architecture/numpilot-architecture-zh.png" alt="NumPilot 系统架构" width="100%">
</p>

主链由六个角色组成——分析器（Analyzer）、规划器（Planner）、缺口补全器（API Gap
Filler）、代码迁移器（Code Migrator）、诊断器（Diagnostics）、迁移报告器（Migration
Reporter）——覆盖分析、迁移、验证三个阶段。解析、能力解析、构建、数值对拍与回执都由
确定性门禁负责；模型可以提出代码，但不能把未经验证的结论升级为通过。

## 实测迁移与加速效果

### 1. 维生素 C：真实量子化学格点密度链路

真实维生素 C 分子（20 原子、92 电子、208 个基函数、81,936 个 DFT 格点），执行
`dm.T @ AO → rho → weighted → electron_count` 链路。

| 统计范围 | CPU 单核 | NPU 设备常驻 | NPU 热态端到端 |
| --- | ---: | ---: | ---: |
| 完整格点密度链路 | 341.750 ms | 13.157 ms（**25.98x**） | 17.419 ms（**19.62x**） |

SCF（约 294.7 s）仍在 CPU 上执行；加速比只覆盖格点密度子阶段。输入为 float32，
`projected` 最大绝对误差 9.54e-6，`rho` 为 1.83e-4，电子数与 CPU 同形式结果完全一致。
该案例是使用 NumPilot 审计运行期的手写迁移（`model_called=false`），不是模型生成算子。

详见：[维生素 C 案例](examples/scientific/gpu4pyscf_vitamin_c/README.md)。

终端录制和框架级六智能体工作台见：[大规模 NumPy 演示](examples/demo/README.md)。其中展示实时智能体面板、模型迁移回执、数值验证，以及严格 NPU 性能口径。

该演示统一使用 [components/asnumpy/](components/asnumpy/) 中的静态 AsNumpy 组件；wheel、源码快照、manifest 和 SBOM 共同构成可复现锁。

### 2. 71 个不含缺口 API 的程序

71 个只使用 AsNumpy 已覆盖 API 的完整 NumPy 程序；其中 66 个有 CPU/NPU 实测数据，
加速比中位数为 11.06x。最快的四个：

| 程序 | 计算内容 | CPU / NPU (ms) | 加速比 |
| --- | --- | ---: | ---: |
| `A009_matrix_chain_2048` | 两个 2048 × 2048 float32 矩阵乘；34.4 GFLOP | 1225.11 / 2.00 | **612.41x** |
| `A007_matrix_product_2048` | 2048 × 2048 float32 矩阵乘；17.2 GFLOP | 608.88 / 1.68 | **361.58x** |
| `B015_fortran_matmul_2048` | 2048 × 2048 Fortran 顺序 float32 矩阵乘 | 634.30 / 1.79 | **355.01x** |
| `H015_real_vecdot_16384x256` | 16,384 × 256 float32；沿两个轴做 vecdot | 91.98 / 0.56 | **164.79x** |

完整清单：[71 例迁移性能表](examples/scaled_benchmark_20261112/BENCHMARK_TABLE.md)。

> **历史证据说明。** 2026-11-12 的 71 例归档作为不可变证据保留，运行环境以归档自身 README 的记录为准；其中的 AsNumpy/CANN 版本不是当前组件基线。当前版本请使用 `examples/demo/`、`examples/scientific/` 与 `components/asnumpy/` 复现。

### 3. 含真实缺口 API 的程序

6 个完整程序，其 NumPy 调用是真实的 API 缺口。每个算子先由配置模型（`llm.json`）生成，
再经过构建、部署、数值验证与准入，然后通过 NumPilot 六智能体链路迁移整个程序。正加速比
的四个：

| 程序 | 缺口 API | 数据规模 | 完整端到端耗时 | CPU / NPU (ms) | 加速比 |
| --- | --- | --- | ---: | ---: | ---: |
| `std` | `np.std` | 2048 × 2048 float32；沿 axis=1 求标准差 | 90.9 min | 15.96 / 0.3718 | **42.92x** |
| `tril` | `np.tril` | 2048 × 2048 float32；默认 k=0 下三角 | 38.4 min | 14.16 / 1.649 | **8.59x** |
| `ediff1d` | `np.ediff1d` | 2048 × 2048 float32 展平；4,194,303 个相邻差分 | 60.7 min | 4.48 / 1.664 | **2.69x** |
| `pad` | `np.pad` | 2048 × 2048 float32；constant 填充 | 47.0 min | 4.117 / 1.724 | **2.39x** |

完整端到端从真实算子生成调用开始，包含构建/修复/部署/数值验证、适配层构建与单测、
完整程序迁移以及 CPU/NPU 基准；迁移命令本身为 6–8 s。

5/6 个程序通过全程序数值对拍；`tril` 的 NumPilot 完整迁移回执保留为失败（`runtime target snapshot is invalid`），
但其独立 strict-NPU benchmark 以 8.59x 通过，不能因此视为完整迁移成功。适配层单测 84 项全部通过。完整表同时保留 `flip`
（0.0042x，CPU view 基线）与 `argpartition`（0.03x，512 元素受 launch 限制）作为
正确性案例。

完整清单：[缺口算子迁移归档](examples/gap_archive_20261112/BENCHMARK_TABLE.md)。

## 贡献与许可证

欢迎提交 Issue 和 Pull Request。可以在仓库中直接提 Issue，或联系
[li-xing@stu.hit.edu.cn](mailto:li-xing@stu.hit.edu.cn)。参见 [CONTRIBUTING.md](CONTRIBUTING.md)
与 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

NumPilot 采用 [Apache-2.0 许可证](LICENSE)。
