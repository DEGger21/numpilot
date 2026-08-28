# NumPilot 可公开证据包

此目录是从完整运行工作盘中筛选出的最小可审计集合，供开发者和评委随源码克隆。它不包含
虚拟环境、上游完整克隆、普通 build 目录或重复失败轮次。所有文件由根目录
`MANIFEST.sha256` 绑定。

## 内容

### `competition-poisson-stage-20260828/`

- 当前更适合评委叙事的 ordinary 应用阶段：Poisson 解析源项与
  `SmoothL1(beta=1)` 未规约逐配点残差图；明确不是完整求解器；
- 9,000,000 个 `float32` 配点，21/21 调用点 DIRECT；实际 `glm-5.3` 迁移、数值门和
  16/16 strict-NPU 性能门全部通过；
- CPU 单核/NPU 各 20 样本的 resident 中位为 `458.698 ms / 178.658 ms`，即
  `2.567×`；保守 `2.471×`，bootstrap 95% 区间 `[2.538×, 2.592×]`；
- 首次单次组件和 NPU 反而慢（诊断 `0.789×`），因此只把重复设备驻留阶段作为优势区；
  回执不把组件和冒充直接端到端；
- 同目录 bootstrap receipt 将 NumPilot `0.3.0a1.dev0`、固定 AsNumpy source build 与
  benchmark runtime hash 闭合；捕获时工作树仍非净，所以不是 RC。

完整解释见
[`competition-poisson-stage-20260828/README.md`](competition-poisson-stage-20260828/README.md)。

### `competition-ordinary-20260828/`

- 当前 ordinary 主路径的开发态真机证据：静态 `DIRECT` 资格化 → 实际 configured
  `glm-5.3` 迁移 → 独立数值验证 → 产品内 strict-NPU resident benchmark；
- Ascend 910B4/CANN 9.0.0，9,000,000 元素 `float32` 合成逐元素链，CPU 与 NPU 分离进程
  固定同一单核，各 20 个样本；16/16 性能门通过；
- resident 中位数为 CPU `616.333 ms`、NPU `76.289 ms`，即 `8.079×`；保守结果
  `7.503×`，bootstrap 95% 区间 `[7.954×, 8.147×]`；
- 不把冷启动、输入准备或最终 D2H 混入 headline；组件和只为 `1.220×` 且不是直接 E2E
  样本；捕获时工作树非净，因此不冒充正式 RC。

完整边界、冻结文件和标准库独立审计命令见
[`competition-ordinary-20260828/README.md`](competition-ordinary-20260828/README.md)。

### `ops-nn-matmul-20260827/`

- 从包含 `numpy.matmul` 缺口的纯 NumPy 小型推理脚本出发，保留迁移前 GapSpec、官方候选
  解析、已配置模型迁移报告、限定范围资格化回执和独立真机执行证据；
- 历史真机结果证明 `_core.matmul` 原生执行与数值，但 2026-08-28 审查发现旧 candidate
  绑定 `libopapi_nn.so`、实际 loader 使用 `libopapi.so`；旧 DSO provenance 已否决，新
  `official-resolution-linkage-v2.json` 目前只到 discovery/candidate，尚未重新资格化；
- Ascend 910B4 上封死宿主回退后原生调用一次，8×16 与 16×4 的 `float32` 结果相对
  NumPy 最大绝对误差为 `6.984919309616089e-10`；不发布性能结论。

完整范围和复核命令见
[`ops-nn-matmul-20260827/README.md`](ops-nn-matmul-20260827/README.md)。

### `demotest-case01-20260825/`

- GLM 生成的 `2048 × 2048`、`float32` 普通 NumPy 脚本及经 NumPilot 门禁的 AsNumpy 迁移产物；
- 源码生成/迁移回执、AST 门禁、CPU/NPU 原始采样、runtime profile/call-count guard、
  ACLNN 日志、设备状态和目录级 SHA-256 manifest；
- 稳定脚本体中位数为 CPU `809.784 ms`、NPU `4.499 ms`，即 NPU 加速 `180.004×`；
  冷进程启动成本另行报告，不混入稳定计算结论；
- 两个大体积数值数组未收入 Git 精选包，JSON 仅保留其哈希和数值比较结果。

完整口径与可信边界见
[`demotest-case01-20260825/README.md`](demotest-case01-20260825/README.md)。

### `heat-diffusion-20260813/`

- `shim/`：离线迁移产物与结构化报告；
- `bench/`：CPU baseline、NPU benchmark、迁移产物和人类可读报告；
- `cpu_baseline.npy` 被保留，因为它是本次 1024×1024 数值对拍的精确基线。

结论、硬件/软件身份、计时边界与复现命令见
[`docs/evidence/heat-diffusion-npu-20260813.md`](../docs/evidence/heat-diffusion-npu-20260813.md)。

### `poisson-ritz-p0-20260821/`

- `qualification-evidence-v5/`：两个定向算子的封存源码、生成/repair provenance、规格、
  构建/测试日志、资格化 JSON、overlay 和 OPP manifest；不包含可重建二进制运行树；
- `runs/poisson-full-iteration-strict-npu-v2/`：117 调用点完整动态一轮；
- `runs/poisson-multiround-strict-npu-v3/`：250 轮、29,250 次公开 API 调用的数值证据；
- `runs/poisson-multiround-strict-npu-v4-move/`：消除包装层深拷贝后重新执行的 250 轮
  strict-NPU 数值与对象生命周期回归；
- `profiler/`：修复前 10 轮 profiler 的配置、导出 CSV、`runtime.db` 和同轮数值证据；
- `profiler-copy-elision/`：已知方向最小探针、修复后 `runtime.db`/`rts_track.db`、新轮
  qualification 身份和机器可审计的修复前后对比。1,170 条 kernel 保持不变，
  `aclrtMemcpy` 从 1,235 降至 35；本证据不发布性能结果。
- `performance-original-scale/`：48×48 内点、49 RBF、3000 轮训练，独立 CPU/NPU worker、
  2 次预热、7 次原始样本、单线程 OpenBLAS 实测、最终数值输出和同规模 profiler 门；
  合格结论为 CPU 约快 `2.67×`。
- `performance-broadcast-optimized/`：同规模、同计时口径的广播/CSE 优化；保存迁移产物、
  3000 轮最终输出、7 次原始样本、profiler DB 和机器审计。每轮调用由 117 降到 101，
  NPU 中位时延下降 `19.71%`，但 CPU 仍约快 `2.879×`。

完整解释见 [`docs/evidence/poisson-ritz-p0-npu-20260821.md`](../docs/evidence/poisson-ritz-p0-npu-20260821.md)
和 [`docs/evidence/poisson-ritz-profiler-p0-20260822.md`](../docs/evidence/poisson-ritz-profiler-p0-20260822.md)。
原始规模性能口径见
[`docs/evidence/poisson-ritz-performance-original-scale-20260823.md`](../docs/evidence/poisson-ritz-performance-original-scale-20260823.md)。
优化前后对照见
[`docs/evidence/poisson-ritz-broadcast-optimization-20260823.md`](../docs/evidence/poisson-ritz-broadcast-optimization-20260823.md)。

## 完整运行树

源码 Git 之外仍保留完整 `numpilot-gap-runs` 与 `numpilot-runs` 冷备份。归档级哈希记录在
`ARCHIVE_MANIFEST.sha256`；这些归档包含约 2 GB 工作盘中的失败轮次、嵌套仓库、venv、
build 与二进制制品，计划作为比赛交付/release 附件单独托管。仅有哈希而没有归档文件时，
不能声称第三方已可下载完整工作盘。

## 校验

在仓库根执行：

```bash
sha256sum -c evidence/MANIFEST.sha256
```

证据报告只证明其明确记录的配置和门禁，不应外推到其他 dtype、shape、硬件、CANN/AsNumpy
版本或完整通用科学计算负载。
