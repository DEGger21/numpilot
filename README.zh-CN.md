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

**环境要求：** 已安装 Python、NumPy 与 AsNumpy。端侧执行需要 CANN 和昇腾 NPU；分析与
shim 验证不需要设备。

```bash
git clone --branch main --depth 1 https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot
python3 -m venv .venv && . .venv/bin/activate
python -m pip install .
python -m numpilot doctor --backend npu
```

迁移并验证一个完整程序：

```bash
numpilot analyze program.py
numpilot migrate program.py --out runs/demo
numpilot verify program.py \
  --migration-report runs/demo/migration/script-migration-report.json \
  --output result --out runs/demo/verification.json
```

如需使用配置模型补全缺口算子，复制空白模板并填入自己的服务信息：

```bash
install -m 600 llm.example.json llm.json
# 编辑 llm.json：base_url、auth_token、model
python -m numpilot configure-llm --check
```

CLI 是唯一的正式执行契约。离线安装与哈希绑定部署见
[安装指南](docs/installation.zh-CN.md)。

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

### 3. 含真实缺口 API 的程序

6 个完整程序，其 NumPy 调用是真实的 API 缺口。每个算子先由配置模型（`llm.json`）生成，
再经过构建、部署、数值验证与准入，然后通过 NumPilot 六智能体链路迁移整个程序。正加速比
的四个：

| 程序 | 缺口 API | 数据规模 | 迁移耗时 | CPU / NPU (ms) | 加速比 |
| --- | --- | --- | ---: | ---: | ---: |
| `std` | `np.std` | 2048 × 2048 float32；沿 axis=1 求标准差 | 7 s | 15.96 / 0.3718 | **42.92x** |
| `tril` | `np.tril` | 2048 × 2048 float32；默认 k=0 下三角 | 8 s | 14.16 / 1.649 | **8.59x** |
| `ediff1d` | `np.ediff1d` | 2048 × 2048 float32 展平；4,194,303 个相邻差分 | 6 s | 4.48 / 1.664 | **2.69x** |
| `pad` | `np.pad` | 2048 × 2048 float32；constant 填充 | 7 s | 4.117 / 1.724 | **2.39x** |

6/6 个程序通过全程序数值对拍，适配层单测 84 项全部通过。完整表同时保留 `flip`
（0.0042x，CPU view 基线）与 `argpartition`（0.03x，512 元素受 launch 限制）作为
正确性案例。

完整清单：[缺口算子迁移归档](examples/gap_archive_20261112/BENCHMARK_TABLE.md)。

## 贡献与许可证

欢迎提交 Issue 和 Pull Request。可以在仓库中直接提 Issue，或联系
[li-xing@stu.hit.edu.cn](mailto:li-xing@stu.hit.edu.cn)。参见 [CONTRIBUTING.md](CONTRIBUTING.md)
与 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

NumPilot 采用 [Apache-2.0 许可证](LICENSE)。
