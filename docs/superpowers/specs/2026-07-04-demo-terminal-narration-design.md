# NumPilot 演示化终端输出 — 设计

- 日期:2026-07-04
- 分支:feat/numpilot-demo
- 目标:一个**简单演示 demo**,证明"真的调用了 asnumpy(真机 NPU)+ 真的迁移了代码",
  并把迁移过程在终端讲清楚。核心是把智能体里**已算出但未输出**的信息透传到终端与报告,
  并补一条"确实用了 asnumpy"的证据。

## 背景

- `numpilot migrate <file>` 已跑通(shim 离线 / npu 真机)。当前 CLI 只打印 5 行:
  状态 / 数值一致 / 单点耗时 / 加速比 / 迁移代码路径。
- 缺失用户关心的:迁移的是**哪个网络**、识别了哪些算子、替换了**多少个** `np.*→ap.*`、
  每个算子 CPU/NPU 分布、以及"真的用了 asnumpy"的凭据。
- 环境已就绪:venv `/home/lixing/np310`(Python 3.12),`asnumpy 0.3.1.dev4`(真机 cp312 wheel),
  已验证 `ap.matmul`/`ap.relu` 在 NPU 上数值与 numpy 一致。
- 性能不达 >1 的根因(910c 未适配 / 健康 numpy 基线)已在记忆中记录,本次**如实呈现**,不做优化。

## 改动点(聚焦,不加新子命令)

### 1. 迁移引擎补替换计数
- `engines/ast_rewrite.py` `_Rewriter`:把 `used_ap: bool` 升级为 `rewrites: dict[str,int]`,
  统计每类替换的次数。key 用可读标签:普通算子用其名(如 `matmul`),relu 模式用 `maximum→relu`。
- `contracts.py` `MigrationResult` 增字段 `rewrites: dict`(默认空 dict);`total_rewrites` 由调用方求和。
- `AstRewriteEngine.rewrite` 返回值不变(仍是 str);计数通过引擎实例上暴露最近一次结果,
  或改由 `MigrationAgent` 读取。**实现取向**:让 `rewrite` 仍返回 str,但引擎在实例上存 `last_rewrites`,
  `MigrationAgent.migrate` 读取填入 `MigrationResult.rewrites`。(保持 `MigrationEngine` 接口稳定。)

### 2. "用了 asnumpy"的执行证据
- `contracts.py` `VerificationReport` 增字段:`backend_module`(asnumpy `__file__`,shim 下为 None)、
  `backend_version`(asnumpy `__version__`,shim 下为 None)、`ap_call_sites`(迁移源码中 `ap.` 调用点计数)。
- `verification.py`:npu 后端下从 `ap_module` 读 `__file__`/`__version__` 回填;`ap_call_sites` 由静态扫描
  迁移源码 AST 得到(`ap.<attr>(...)` 调用数)。shim 后端回填 shim 标识即可(证明是仿真而非真机)。

### 3. 终端叙事输出
- `__main__.py` `_cmd_migrate`:抽出一个 `_render_terminal(res)`,按分段打印:
  ```
  ── NumPilot 迁移 · <网络名> ─────────────
  [分析] N 处 numpy 调用 · M 个算子 · 全部在覆盖集 ✓ | 缺口: ...
  [迁移] 引擎=ast · 替换 K 处 np.*→ap.*
         matmul ×2  maximum→relu ×1  add ×2 ...
  [卸载] NPU: ...   CPU: ...
  [执行] 后端=npu · asnumpy <ver> (<path>) ✓ 真机 | shim 仿真
  [校验] 数值一致 ✓  max_rel_diff=..  (tol=..)
  [性能] <n> 元素   numpy@CPU ..ms   asnumpy@npu ..ms   ..x
  [产物] <migrated.py> · <report.md>
  ```
  - 网络名 = 源文件名去扩展名。
  - unsupported 场景:打印 `[分析]` 命中缺口 + 缺口算子清单,提前返回,不进入后续段。
  - 保留退出码语义(pass=0 / fail=2 / unsupported=1 / env 拦截=3)。

### 4. 报告同步
- `orchestrator._render_md`:补"替换算子清单"与"执行证据"两节,字段与终端一致;
  `_maybe_write` 的 summary 增加 `rewrites` / `total_rewrites` / `backend_module` / `backend_version` /
  `ap_call_sites`。JSON 向后兼容(仅新增字段)。

## 不做(YAGNI)

- 不 push 到 gitcode;不做 Phase 5/6 额外产物;不加 list-ops / 批量 / --json 子命令;
  不碰性能优化;不改覆盖集与算法。

## 验证

- `--backend shim` 离线:`mlp_forward`、`big_gemm`、`unsupported_demo` 三个 example 输出符合分段格式。
- `--backend npu` 真机:`mlp_forward`、`big_gemm` 跑通,`[执行]` 段显示真实 asnumpy 版本与路径、数值一致。
- 现有 23 个 pytest 全过;新增测试:替换计数(`rewrites` 内容)、`ap_call_sites` 计数、终端渲染不抛异常。

---

## 追加(2026-07-04):性能对比拆成两个独立程序

### 动机(实测根因)
本机 numpy=**scipy-openblas**(OpenMP 线程后端),`matmul` 默认吃满全部 **640 核**:
同一 2048³ matmul,1 线程 370ms → 640 线程 15.8ms(23×)。原始"numpy@CPU 0.29×"
几乎全部来自 640 核堆料,不是 NPU 慢——限 1 核后 NPU 反而快 8.45×。**这修正了之前
"910c 未适配所以打不过 numpy"的归因:真正主因是 CPU 基线被 640 核 OpenBLAS 加持。**

### 为什么必须分进程
单进程内用环境变量限核会**互相污染**:scipy-openblas 的 `OPENBLAS_NUM_THREADS` 会调
`omp_set_num_threads()` 改全进程 OpenMP,而 CANN 主机侧也用同一 OpenMP,导致设 1 核时
NPU 也被拖慢 10×(133ms→1239ms)。故把两次测量拆到**两个独立进程**,各设各的线程,互不干扰。

### 设计
- `bench-cpu <file> --out DIR`:纯 numpy 进程,`__main__` 早期(import numpy 前)全局钉
  1 核(此进程不碰 NPU,钉核安全);跑原始 forward 计时,存 `cpu_baseline.{npy,json}`。
  `--cpu-threads N`(默认 1;0/all=全核对照)。
- `bench-npu <file> --out DIR`:放开所有限制,迁移→NPU 上跑 asnumpy 计时;就地用 numpy
  满线程参考做数值校验;读 `cpu_baseline.json` 算 加速比=t_cpu(1核)/t_npu(满线程),打印 asnumpy 真机证据。
- `migrate`:回归"迁移证明"职责(分析/迁移/正确性/asnumpy 证据),不再自测性能;
  `[性能]` 段与 report.md 改为指向上面两条 bench 命令。`VerificationAgent.verify(measure=False)`。
- 新模块 `numpilot/bench.py`;新增 bench-cpu/parse_cpu_threads 测试;32 pytest 全过。

### 实测(big_gemm 2048³,真机)
`bench-cpu` numpy@1核 1543ms;`bench-npu` asnumpy@npu 182ms(未被拖慢)→ **8.45× asnumpy 更快**,数值 0.00e+00。
mlp_forward(640 元素)0.44×(小负载 NPU 启动开销占主导,诚实边界)。
