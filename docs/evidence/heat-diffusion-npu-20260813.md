# NumPilot 二维热扩散 strict-NPU 实测留档

> 实验时间：2026-08-13 UTC；归档时间：2026-08-14 UTC。未使用 Docker。
> 公开精选证据：[`evidence/heat-diffusion-20260813/`](../../evidence/heat-diffusion-20260813/)；
> 历史完整运行根：`/home/ma-user/work/numpilot-runs/heat-diffusion-20260813/`。

## 1. 结论

NumPilot 成功把一个纯 NumPy 编写的二维热扩散单步网络迁移为
`whole_forward_npu`：7 个 NumPy 调用全部改写，strict forward 静态审计、两条运行时
callsite 门禁、三层数值比较和 CPU baseline 一致性全部通过。实际 AsNumpy capability 均为
`native_npu`，没有启用或使用 CPU fallback。

当前保留主证据对应的结果为：单线程 NumPy CPU 中位时延 `15.552298980765045 ms`，NPU
中位时延 `14.424319961108267 ms`，`performance_valid=true`，比值
`1.07819980579314×`，即本次合格口径下 NPU 快约 `1.08×`。所有数值比较的最大绝对误差和
最大相对误差均为 `0.0`。

同一 CPU baseline 上此前还执行过一次 NPU 重复测量，控制台结果为 `13.67651391774416 ms`
和 `1.1371537421233642×`。该次 JSON 随第二次运行被同名输出覆盖，没有保留完整原始产物，
因此只作为波动观察，不作为本档案的主证据。两次结果表明当前收益约为 `1.08–1.14×`，优势
较小，不应外推为稳定或通用的 NPU 加速结论。

## 2. 科学计算负载与迁移边界

示例源码为 `examples/heat_diffusion_forward.py`。它计算带周期边界条件的二维热方程一次显式
Euler 更新，采用五点有限差分：

```text
T_next = T + alpha_dt * (N + S + W + E - 4 * T)
```

- 网格：`1024 × 1024`；
- dtype：`float32`；
- 初始场：二维 Gaussian 温度场；
- `alpha_dt=0.20`，满足该归一化二维显式格式的稳定条件 `alpha_dt <= 0.25`；
- `forward()` 中共有 `add×4`、`multiply×2`、`subtract×1`；
- `build_inputs()` 在 CPU 侧用 `np.roll` 生成四个周期邻居，并物化为 C-contiguous 数组。

这是一层真实的 PDE 单步计算网络，不是完整多步求解器。NumPilot 当前只迁移顶层
`forward()`，尚不能把 `np.roll`、切片、下标或时间步循环纳入 strict whole-forward NPU 岛，
所以邻域采样留在 CPU `build_inputs()` 中。该边界是当前系统能力限制，不能把本次结果描述成
完整 PDE 求解器端到端加速。

## 3. 代码与环境身份

| 项目 | 实测值 |
|---|---|
| OS / 架构 | Ubuntu 22.04.5 / aarch64 |
| CPU | Kunpeng-920，192 logical CPUs |
| NPU 独立观测 | device 4，Ascend 910B4，Health OK |
| `npu-smi` | 25.5.1 |
| CANN | 9.0.0，`/usr/local/Ascend/cann-9.0.0` |
| Python | 3.10.20 |
| CPU baseline NumPy | 1.26.4 |
| NPU 评测进程 NumPy | 2.2.6 |
| AsNumpy | `0.3.1.dev4+g1929a3ba4` |
| AsNumpy 模块 | `/home/ma-user/work/numpilot-runs/heat-diffusion-20260813/audited-site/asnumpy/__init__.py` |
| capability audit ID | `asnumpy-0.3.1.dev4+g1929a3ba4-source-1929a3ba4-v1` |
| Git branch / HEAD | `feat/numpilot-demo` / `694e3f99cbbb849bfb9c967471b3a448c44211e8` |

评测使用原始审计 wheel：

```text
/home/ma-user/work/asnumpy/dist/asnumpy-0.3.1.dev4+g1929a3ba4-cp310-cp310-linux_aarch64.whl
sha256=cd65ce9b02ec75e0fad4a3d0e3b9797f76a685a7a673d6bbef6cd91cc8e23f0a
```

wheel 被离线安装到独立 `audited-site`；安装后的 `RECORD` 共 32 项，大小和哈希均核验通过，
native `_core.so` SHA-256 为
`ab71b0041f872bd25f4581a9f269414af9accd37f83e4ad8d99b3e713d63ff53`。
同一证据根下的 `audited-venv` 是最初失败的准备尝试：它使用 NumPy 1.26.4 且缺少
`loguru`，不是本次 NPU 运行环境，也不能用于复现。实际运行由
`asnumpy-runtime-20260813/venv` 提供 NumPy 2.2.6 和 `loguru` 0.7.3，再通过
`PYTHONPATH` 优先加载上述 `audited-site`。

实验发生在有未提交修改的工作树上，示例源码当时也是未跟踪文件；因此 Git HEAD 不能单独
重建本次代码。可复现身份以本文件列出的源码/迁移产物哈希和持久证据目录为准。

## 4. 迁移与真实性门禁

| 门禁 | 结果 |
|---|---|
| 静态分析 | `migratable=true`，`unsupported=[]` |
| placement | `add/multiply/subtract -> NPU`，策略为 `whole_forward_npu` |
| 预期/实际改写 | 均为 `add×4, multiply×2, subtract×1` |
| rewrite integrity | `true` |
| strict forward surface | 通过，无 residual NumPy/host escape |
| capability provenance | 三个算子均为 `native_npu`、`strict_npu=true` |
| CPU fallback | `allow=false`、`used=false`、fallback 列表为空 |
| kernel 运行路径 | 实际捕获 `add×4, multiply×2, subtract×1` |
| artifact 运行路径 | 实际捕获 `add×4, multiply×2, subtract×1` |
| runtime trace / target integrity | 均通过 |
| 迁移产物自身 `build_inputs()` | 已执行 |

数值门禁使用 `atol=1e-5`、`rtol=1e-4`：

| 比较层 | 元素数 | max abs | max rel | 结果 |
|---|---:|---:|---:|---|
| 原始/迁移输入生成 | 7,340,032 | 0.0 | 0.0 | 通过 |
| 同一规范输入上的 forward | 1,048,576 | 0.0 | 0.0 | 通过 |
| 迁移产物 `build_inputs + forward` | 1,048,576 | 0.0 | 0.0 | 通过 |
| 保存的 CPU baseline 输出 | 1,048,576 | 0.0 | 0.0 | 通过 |

## 5. 性能口径

- CPU baseline 明确请求 `OPENBLAS_NUM_THREADS=1`；该 forward 只有逐元素运算，不调用
  BLAS。报告中的线程数来自环境值，不是 `threadpoolctl` 实测值；
- 当前代码快照的计时器预热 3 次、测量 10 次并取排序后的中位样本；原始逐轮样本没有保存；
- NPU 输入在计时前已驻留设备，排除 H2D；
- 每轮计时都包含递归 D2H 回传和同步；
- 因此该口径既不是 kernel-only，也不是完整 end-to-end；
- 仅因 `performance_valid=true`、strict-NPU、数值和 baseline 门禁均通过，才引用
  `1.07819980579314×`。

## 6. 实际执行命令

先把审计 wheel 安装到隔离目录；没有改写原 AsNumpy 仓库、系统 Python 或系统 OPP：

```bash
/home/ma-user/work/numpilot-gap-runs/asnumpy-runtime-20260813/venv/bin/python \
  -m pip install --no-index --no-deps \
  --target /home/ma-user/work/numpilot-runs/heat-diffusion-20260813/audited-site \
  /home/ma-user/work/asnumpy/dist/asnumpy-0.3.1.dev4+g1929a3ba4-cp310-cp310-linux_aarch64.whl
```

离线 shim 迁移与三层语义验证：

```bash
cd /home/ma-user/work/numpilot
/home/ma-user/miniconda3/bin/python -m numpilot migrate \
  examples/heat_diffusion_forward.py \
  --backend shim --engine ast --skip-env-check \
  --out /home/ma-user/work/numpilot-runs/heat-diffusion-20260813/shim
```

独立单线程 CPU baseline：

```bash
cd /home/ma-user/work/numpilot
/home/ma-user/miniconda3/bin/python -m numpilot bench-cpu \
  examples/heat_diffusion_forward.py \
  --out /home/ma-user/work/numpilot-runs/heat-diffusion-20260813/bench \
  --cpu-threads 1
```

真实 NPU 评测从空环境启动，避免继承环境把 CANN `devlib` stub 排在真实 driver 前面；没有
传入 `--allow-cpu-fallback`：

```bash
env -i \
  PATH=/home/ma-user/work/numpilot-gap-runs/asnumpy-runtime-20260813/venv/bin:/home/ma-user/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  PYTHONPATH=/home/ma-user/work/numpilot-runs/heat-diffusion-20260813/audited-site \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 LANG=C.UTF-8 \
  bash --noprofile --norc -c '
    source /usr/local/Ascend/cann-9.0.0/set_env.sh
    cd /home/ma-user/work/numpilot
    exec /home/ma-user/work/numpilot-gap-runs/asnumpy-runtime-20260813/venv/bin/python \
      -B -m numpilot bench-npu examples/heat_diffusion_forward.py \
      --out /home/ma-user/work/numpilot-runs/heat-diffusion-20260813/bench \
      --skip-env-check
  '
```

`--skip-env-check` 是因为当时 NumPilot 的 `npu-smi` 文本解析器错误报告“存在命令但没有
device info”；同一清洁环境中的独立 `npu-smi` 已看到 910B4、Health OK，AsNumpy 导入和
真实 ACLNN Add/Mul/Sub 随后也成功。该参数只跳过有缺陷的预检，不跳过 capability、strict
surface、运行时 callsite、target integrity、数值或 fallback 门禁。报告中的
`soc_version=unknown` 也是该环境探测缺口，不能用来否定或替代独立硬件观测。

复现时应改用新的 `--out` 目录，避免覆盖本档案的主证据。

## 7. 证据文件与 SHA-256

仓库内文件：

```text
6f83b11ae500b2201b6bd1e416f1b67e935bf8d147d80a385f8c2eb605e2bc9f  examples/heat_diffusion_forward.py
68ea47d6d0f7ac8e2a96081a6cbc58f860d4e9764ed253f81c858651918c2997  evidence/heat-diffusion-20260813/migration_report.md
```

持久证据根：`/home/ma-user/work/numpilot-runs/heat-diffusion-20260813/`

```text
8f484b1833e7cc3b7833a9b8366a2725c96e134a0e416c1a90a3f67663699836  shim/report.json
2bb8661b996102c931fb94801347cc7615f6e98880e1a5a95f8bd9629013b06e  shim/report.md
f1713c39aa9865b781dd0cb53dcab3c83e3278b9cb833936e491a36571f1b2bf  shim/migrated.py
d23ab39bde510ec3bb47b198a1e03b3107e556d9e08b365e71b9150fc407f89f  bench/cpu_baseline.json
2614bb2fa529c84ab355ba9d445c1527d8aee0fb752dfcae1162935f23b1e440  bench/cpu_baseline.npy
f1713c39aa9865b781dd0cb53dcab3c83e3278b9cb833936e491a36571f1b2bf  bench/migrated.py
6710eea7776aedd201a16c5b270c7411c14ae11173b26b5b849ad7be91432119  bench/npu_benchmark.json
4454626f1f497beeac7684086da48d2e039ce34ffbd5798a2fb8b63714531fcf  bench/benchmark_report.md
```

shim 与 NPU 目录中的 `migrated.py` 字节一致；`npu_benchmark.json` 内记录的 source 和
migrated SHA-256 也分别与当前源码和产物一致。

## 8. 证据边界与后续改进

- 当前真实性证据是版本绑定的 capability/source audit、strict 静态 surface、Python API
  runtime trace 和 exact callable integrity；它还不等价于 profiler/kernel trace；
- 没有保存 ACL profiler 原始目录、kernel 时间线、传输字节或逐轮 benchmark 样本；
- 主 JSON 没有记录 NumPy 版本、设备号，且 `soc_version` 为 `unknown`；这些值来自本轮独立
  环境核对，应在后续 schema 中直接落盘；
- `npu-smi` parser 对本机输出存在 false negative，应修复后取消 `--skip-env-check`；
- 当前 7 个逐元素 kernel 没有图融合。若要获得更明显收益，应先实现 profiler 证据，再基于
  实际 kernel/launch 时间评估融合或更大工作负载，不能仅凭本次小幅收益推断。
