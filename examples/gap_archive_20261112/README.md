# 含真实缺口算子的完整程序迁移归档（2026-11-12）

本目录保存 **6 个完整 NumPy 程序 → 6 个完整 AsNumpy 程序**的迁移证据链。
每个 `program.py` 都包含 `build_inputs()` 与 `forward()`；迁移后的 `migrated.py`
保持同一程序结构，只把 NumPy 调用替换为 AsNumpy 调用。六个程序各含 1 个真实
缺口 API，缺口算子的生成/补全耗时见 `BENCHMARK_TABLE.md` 的第二张表。

| # | NumPy 程序 | AsNumpy 程序 | 数据规模 | 替换的 API | 迁移耗时 | CPU / NPU (ms) | 加速比 |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | `cases/std/program.py` | `cases/std/migrated.py` | 2048 × 2048 float32；沿 axis=1 求标准差 | `np.std` | 7 s | 15.96 / 0.3718 | **42.92x** |
| 2 | `cases/flip/program.py` | `cases/flip/migrated.py` | 2048 × 2048 float32；沿 axis=1 翻转 | `np.flip` | 8 s | 0.007381 / 1.756 | 0.0042x（口径见下） |
| 3 | `cases/ediff1d/program.py` | `cases/ediff1d/migrated.py` | 2048 × 2048 float32 展平；4,194,303 个相邻差分 | `np.ediff1d` | 6 s | 4.48 / 1.664 | **2.69x** |
| 4 | `cases/pad/program.py` | `cases/pad/migrated.py` | 2048 × 2048 float32；constant 填充；输出 2050 × 2052 | `np.pad` | 7 s | 4.117 / 1.724 | **2.39x** |
| 5 | `cases/tril/program.py` | `cases/tril/migrated.py` | 2048 × 2048 float32；默认 k=0 下三角 | `np.tril` | 8 s | 14.16 / 1.649 | **8.59x** |
| 6 | `cases/argpartition/program.py` | `cases/argpartition/migrated.py` | 32 × 16 float32；沿 axis=1、kth=3 分区 | `np.argpartition` | 7 s | 0.01049 / 0.3489 | 0.03x（口径见下） |

每个案例都**完全走 NumPilot 流水线**，不是手工替换后补一份报告：

1. `numpilot complete-gaps --provider llm --allow-custom-operator-generation`
   分析缺口、调用真实模型（`llm.json`，调用前获批）生成 Ascend C 算子草稿。
2. 构建 → 修复 → 部署 → 数值验证，生成 `custom-opp-provenance.json`。
3. 用算子 vendor 构建 AsNumpy overlay wheel（不修改 AsNumpy 主干），运行算子单测。
4. 生成准入证据 `admission.json`，由 NumPilot 在迁移时校验 admission id。
5. `python -m numpilot migrate --engine ast --backend npu` 对完整 `program.py`
   执行 6 智能体链路，生成完整 `migrated.py`：
   Analyzer → Planner → API Gap Filler → Code Migrator → Diagnostics → Migration Reporter。
6. strict-NPU 执行完整程序、逐输出对拍、全程序 CPU/NPU 基准。

## 目录

```
BENCHMARK_TABLE.md            6 例完整程序性能表与缺口算子明细
MANIFEST.json                 文件哈希与逐例元数据
cases/<case>/program.py       完整 NumPy 程序（数值 oracle）
cases/<case>/migrated.py      NumPilot 生成的完整 AsNumpy 程序
cases/<case>/case.json        单例指标、准入 id、基准与注意事项
operators/<case>/draft/       模型生成的原始草稿（保留未修改）
operators/<case>/final/       最终可构建/部署的算子源码（含确定性修复）
operators/<case>/adapter/     生成的 AsNumpy 适配层（header/csrc/pybind/wrapper/test）
evidence/<case>/              缺口、生成、准入、provenance、迁移、基准证据（已脱敏）
```

## 关键数字

- 完整程序迁移：**6/6** 生成完整 `migrated.py`，全程序逐输出对拍通过
- 缺口算子生成：**14.5–16.9 min**（均值 15.3 min，6/6 为真实模型调用）
- 生成+补全（模型 + 构建修复 + 部署 + 数值验证）：**32.1–57.2 min**（均值 41.9 min）
- NumPilot 迁移命令（完整程序级）：**6–8 s**（6/6 通过，6 智能体链路完整执行）
- 适配层单测：**84 passed**
- 端到端（模型调用到基准完成）：**38.2–90.9 min**（均值 58.4 min）
- 数值对拍：**6/6 通过**（std 最大绝对误差 2.38e-7，在 atol=1e-5/rtol=1e-4 内；其余为 0）

## 口径与边界（PPT 可直接引用）

- **“生成”是模型调用，“补全”是确定性工程修复**。模型草稿与最终代码分开归档：
  `operators/<case>/draft/` 是模型原样输出，`operators/<case>/final/` 是修复后
  实际构建/部署的版本。修复次数见 `case.json` 的 `build_attempts`。
- **`flip` 的 CPU 基线是 view**：`numpy.flip` 返回 O(1) 视图，而 NPU 算子真实
  搬运 4 Mi 元素；0.0042x 不是等价工作量下的性能结论，不要单独引用。
- **`argpartition` 是正确性案例，不是加速案例**：512 元素规模由 launch + D2H
  同步主导（0.349 ms vs CPU 0.010 ms）。它的价值是最终 kernel 逐行复刻
  numpy 2.5.4 `introselect_`（arg=true），在 5256 组对抗样例上与
  `np.argpartition` 完全一致，并通过 19 个单测。**不要把 0.03x 放进加速比宣传口径。**
- **`std` 是 2048 元素归约**，42.92x 反映的是小规模归约在 NPU 上的固定开销优势，
  不代表大规模吞吐；大规模数据请引用 71 例无缺口基准。
- 加速比口径：CPU 单核 `forward()` 中位耗时 ÷ NPU 设备常驻 `forward()` 中位耗时
  （含结果回传，不含 H2D 与输入构造）。

## 证据与安全

- 所有 JSON 已去除模型私有字段（token 计数、prompt/response 哈希、workspace 路径）。
- 所有文本中的服务器私有绝对路径已替换为相对路径或 `<server-home>`。
- 未归档 `llm.json`、私钥、原始模型响应或 prompt 明文。
- 服务器原始证据保留在完整运行目录中；本目录是面向评审/PPT 的脱敏副本。
- 确定性修复前的中间备份不进入公开归档；修复轮次记录在 `case.json` 的
  `build_attempts` 与 evidence 的 `build-attempts.json` / `timeline.log`。
  模型原始草稿始终以 `operators/<case>/draft/` 为准。

## 相关链接

- [6 例完整性能表](BENCHMARK_TABLE.md)
- [71 例无缺口迁移归档](../no_gap_archive_20261112/README.md)
- [71 例大规模迁移性能表](../scaled_benchmark_20261112/BENCHMARK_TABLE.md)
