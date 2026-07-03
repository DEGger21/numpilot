# NumPilot · LLM 引擎手动运行指南(临时)

> 临时操作备忘。用 `--engine llm` 跑真实模型迁移,并对比不同模型的产出。
> 凭据放在 `.env.llm`(已被 gitignore),本文件不含任何 token。

---

## 第 1 步:填凭据

编辑 `/home/lixing/numpilot/.env.llm`,填好三个必填项(缺任一会自动回退 AST 引擎):

| 变量 | 必填 | 说明 |
|------|------|------|
| `NUMPILOT_LLM_BASE_URL` | ✅ | OpenAI 兼容 endpoint,一般带 `/v1`(代码自动拼 `/chat/completions`) |
| `NUMPILOT_LLM_AUTH_TOKEN` | ✅ | 鉴权 token(以 `Authorization: Bearer` 发送) |
| `NUMPILOT_LLM_MODEL` | ✅ | 比赛限定的模型名,**每次换模型改这一行** |
| `NUMPILOT_LLM_PROVIDER` | ⬜ | 可选,仅作标签 |

---

## 第 2 步:进目录 + 加载凭据 + 跑

```bash
cd /home/lixing/numpilot
set -a && . ./.env.llm && set +a          # 加载环境变量到当前 shell(本终端后续命令都可用)

# 核对变量确实加载了(token 只显示前 6 位)
echo "base=$NUMPILOT_LLM_BASE_URL  model=$NUMPILOT_LLM_MODEL  token=${NUMPILOT_LLM_AUTH_TOKEN:0:6}..."

# 用 LLM 引擎迁移
python -m numpilot migrate examples/mlp_forward.py --engine llm --out out_llm/
```

---

## 第 3 步:确认真走了 LLM(没静默回退 AST)

```bash
python -m numpilot migrate examples/mlp_forward.py --engine ast --out out_ast/
diff out_ast/migrated.py out_llm/migrated.py
```

- **有差异** → LLM 确实产出了自己的版本 ✅
- **完全相同 / 无输出** → 很可能凭据没配对、回退了 AST,回到第 1 步检查

---

## 第 4 步:换模型对比

改 `.env.llm` 里的 `NUMPILOT_LLM_MODEL` 一行,重新加载再跑到不同目录:

```bash
set -a && . ./.env.llm && set +a          # 改完模型后重新加载
python -m numpilot migrate examples/mlp_forward.py --engine llm --out out_模型名/
```

产物在各自 `out_*/` 目录下(`migrated.py` / `report.md` / `report.json`),横向对比 `migrated.py` 即可看模型差异。

---

## 排错速查

| 现象 | 原因 | 处理 |
|------|------|------|
| 状态 pass 但 migrated 与 AST 完全一样 | 三个变量缺一 / endpoint 打错 → 回退 AST | 查第 2 步 echo 输出,确认 base、model 非空 |
| 401 / access denied | token 无效或无写权限 | 换 token |
| endpoint 相关报错 | base_url 没带 `/v1` 或多尾斜杠 | 给到 `.../v1` 即可,代码自动拼 `/chat/completions` |
| 卡住很久 | 网络到网关不通 | 超时 60s 后自动回退 AST |

---

## 说明

- **验证始终在 shim 后端跑**:不管哪个模型,只要产出的迁移代码数值正确,`状态` 就是 `pass`;若模型把逻辑改错,会 `fail` 或 `max_rel_diff` 变大——这正是要观察的模型效果差异。
- `.env.llm` 含明文 token,别外传;`out*/`、`.env.llm` 均已 gitignore。
- 用完可删本文件:`rm /home/lixing/numpilot/RUN_LLM_TEMP.md`

---
---

# 📌 会话进展快照(asnumpy 真机安装 + numpilot 环境自检 + demo 重设)

> 追加于 2026-07-03。用户需切换网络环境,此处落盘当前进展与续跑步骤。
> 上面的 LLM 引擎指南与本节无关,勿删。

## 总目标(用户本轮三项需求)

1. **numpilot 加环境自检**:用框架前先检查本机环境(参考 cannbot-skill 的 `ascendc-env-check/scripts/check_env.sh` 做法)。
2. **安装真实 asnumpy**:从 upstream `https://gitcode.com/cann/asnumpy` 拉取最新并按其 README 安装(替换掉现在的 shim 假后端)。
3. **重设 demo**:纯 numpy 搭网络 → 全 CPU 跑一遍并计时 → 迁移到 asnumpy → NPU 跑 → 对比**精度(max_rel_diff)**与**时间(加速比)**;数据量要**非常大**(越大 asnumpy 越有优势)。

## 已锁定的设计决策(经 AskUserQuestion 确认)

- **demo 网络形态** = **大 GEMM 主导网络**(如若干 4096×4096 级矩阵乘;matmul 最能体现 NPU 优势)。
- **migrate 前置自检** 发现 npu 后端缺 asnumpy/CANN 时 = **直接拦截并退出**(清晰报错 + 非零退出码,留 `--skip-env-check` 逃生口;shim 后端仅软告警)。
- **Python 版本策略** = **为 asnumpy 上 Python 3.10+ 栈**(用户选定,否决了"打 5 行补丁适配 3.9")。

## 环境事实(已探明)

| 项 | 值 |
|----|----|
| 系统 Python | **仅 3.9.9**(`/usr/bin/python`);dnf 无 3.10+ 包 |
| CANN | 8.5.0,`ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest`;**编译/运行前先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`** |
| 工具链 | gcc 14.3.1、cmake 4.2.3(满足) |
| NPU | 存在(npu-smi 显示多颗 Ascend910) |
| numpy/torch | 3.9 下 numpy 1.26.0、torch 2.8.0+cpu、torch_npu 2.8.0.post4 |
| **numpilot 依赖** | **不 import torch**(已确认),迁移流程只需 numpy + asnumpy → 3.10 环境很轻 |
| fmt/spdlog | **已装** `fmt-devel 8.0.1` + `spdlog-devel 1.11.0`(CMake `find_package(... REQUIRED)` 隐藏系统依赖,README 未提;已持久化) |
| uv | **已装** `/usr/local/bin/uv`(0.11.26);⚠️ `python -m uv` wrapper 找不到 binary,**直接用 `/usr/local/bin/uv`** |

## asnumpy 安装:已做 & 当前卡点

- 已从 upstream 干净 `git clone --recursive` 到 **`/home/lixing/asnumpy`**(master `1929a3b`,version `0.3.1.dev4`,无子模块)。
- 已按 README pip 路径编出 wheel:`/home/lixing/asnumpy/dist/asnumpy-0.3.1.dev4+g1929a3ba4-cp39-cp39-linux_aarch64.whl`(**cp39,3.10 路径下作废,需换 3.12 重编**)。
- 试装进系统 3.9(`--ignore-requires-python`)→ `import asnumpy` 可以,但 **`ap.ones()` 运行期报错**:
  `_types.py:37` `DTypeLike = np.dtype | str | type | None`(PEP 604 联合类型,**需 Python 3.10+**)。
  → 全包共 5 行此类语法(`_types.py` 第 37/39/41/43/45),无 match/case。**这就是必须上 3.10 的根因**(pyproject `requires-python>=3.10` 其实是对的,README 徽章"3.9+"是错的)。
- ⚠️ 系统 3.9 里可能残留一个装坏的 asnumpy,续跑时先 `python -m pip uninstall -y asnumpy` 清掉,避免干扰。

## ▶️ 续跑步骤(网络恢复后,按序执行)

```bash
UV=/usr/local/bin/uv
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 1) 拉 CPython 3.12(python-build-standalone,需联网到 astral 镜像)
$UV python install 3.12
$UV python list | grep 3.12          # 确认已就位

# 2) 建 3.12 venv(放框架同级,便于 numpilot 用)
$UV venv --python 3.12 /home/lixing/np310
source /home/lixing/np310/bin/activate
python --version                     # 应为 3.12.x

# 3) 干净按 README 装 asnumpy(3.12 下 requires-python 满足,无需补丁/ignore)
cd /home/lixing/asnumpy
rm -rf dist build                    # 清掉 cp39 旧产物
pip install --upgrade pip setuptools wheel
pip install build
python -m build                      # 重新编 cp312 wheel(fmt/spdlog 系统已装)
pip install dist/*.whl

# 4) README 冒烟验证(这次应真正通过)
python -c "import asnumpy as ap; a=ap.ones((1000,1000),dtype=ap.float32); print(a.shape)"
# 期望: (1000, 1000)

# 5) 把 numpilot 装进同一个 venv
pip install -e /home/lixing/numpilot   # 或直接 cd /home/lixing/numpilot 后跑
python -c "import numpilot, numpy; print('numpilot ok')"

# 6) 真机端到端验证
cd /home/lixing/numpilot
python -m numpilot migrate examples/mlp_forward.py --backend npu --out out_npu/
# 期望: 状态 pass、数值一致 True;加速比此时才是"真 NPU vs numpy"
```

若第 1 步 `uv python install` 因网络失败:备选 = 源码编译 CPython 3.12,或装 miniforge(aarch64)。

## 之后待做(Part 2/3,尚未写码 —— 已完成 brainstorming,决策见上)

- **Part 2 环境自检**:新增 `numpilot/env_check.py`(检查 Python、numpy、CANN Toolkit/set_env、npu-smi/设备、**asnumpy 可导入性**,累计 ERROR/WARN);接入 ① 新子命令 `python -m numpilot doctor`;② `migrate` 前置自检,`--backend npu` 缺依赖则**拦截退出**,加 `--skip-env-check`。参考 `/home/lixing/cannbot-skills/ops/ascendc-env-check/scripts/check_env.sh` 的分项+汇总风格。
- **Part 3 大 GEMM demo**:新增大规模纯 numpy 示例(仅用覆盖集算子,见 `numpilot/coverage.py`);增强 `numpilot/agents/verification.py` 使真机路径**对输入 `to_device` + 计时时同步**(现状是直接把 numpy 喂给迁移侧、`perf_counter` 裸测,NPU 异步下测不准);报告里显式并列 "numpy@CPU 时间 vs asnumpy@NPU 时间"。
- **关键代码位置**:`numpilot/agents/verification.py`(测速/精度)、`numpilot/backends/{shim,npu}.py`(后端)、`numpilot/backends/base.py:23`(`NUMPILOT_BACKEND` 默认 shim)、`numpilot/orchestrator.py`(流程编排)、`numpilot/__main__.py`(CLI 入口)。

---
---

# 📌 会话进展快照 #2(asnumpy 真机装好 + Part 2 完成 + Part 3 进行中)

> 追加于 2026-07-03,用户再次换网络前落盘。**关键:asnumpy 已装到本地磁盘,后续全部工作(改代码、跑真机)不再依赖网络。**

## ✅ 已完成:asnumpy 真机安装(Part 1 收尾)

- CPython 3.12.13 已装(`/usr/local/bin/uv python install 3.12`),venv 在 **`/home/lixing/np310`**。
- asnumpy cp312 wheel 编译成功:`/home/lixing/asnumpy/dist/asnumpy-0.3.1.dev4+g1929a3ba4-cp312-cp312-linux_aarch64.whl`,已 `uv pip install`。
- numpilot 已 `uv pip install -e` 进同一 venv。拉入 **numpy 2.5.0** + loguru 0.7.3。
- 冒烟通过:`ap.ones((1000,1000),dtype=ap.float32)` OK;`python -m numpilot doctor --backend npu` **全绿**。

### 环境激活(每次新终端必做)
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/lixing/np310/bin/activate      # Python 3.12.13
# ⚠️ uv venv 无 pip,装包用 /usr/local/bin/uv pip install ...
```

## 🔑 asnumpy 真实 API 与约束(实测,续跑必读)

| 事实 | 细节 |
|------|------|
| numpy→device | `ap.ndarray.from_numpy(np_arr)` → NPUArray;`dev.to_numpy()` → numpy。NpuBackend 已这么写,正确 |
| 算子形态 | `ap.matmul/add/maximum/...` 作用于 **NPUArray**;`ap.array/asarray/from_numpy` 顶层**不存在**(只有 `ap.ndarray.from_numpy`) |
| **标量不支持** | `ap.add(dev, 1.0)` / `ap.multiply(dev, 2.0)` **FAIL**(要求两侧都是 NPUArray)。`ap.matmul(numpy,numpy)` 能自动转,但混合 `(NPUArray, numpy)` **FAIL** |
| relu 可用 | `ap.relu(dev)` OK 且数值正确;`ap.maximum(dev,dev)` OK,但 `ap.maximum(dev, 0)` **FAIL**(标量 0) |
| 同步 API | 有 `ap.init/finalize/set_device`;可靠同步点 = `dev.to_numpy()`(强制 D2H completion) |

## ✅ 已完成:Part 2 环境自检

- 新增 **`numpilot/env_check.py`**:`Env.check(backend)` → `EnvReport`(逐项 CheckItem + error/warn 计数)。检查 Python≥3.10 / numpy / CANN Toolkit(ASCEND_TOOLKIT_HOME+OPP) / npu-smi 设备 / **asnumpy 可导入且可运行**。硬依赖严重度随后端:npu→ERROR 拦截,shim→WARN 软告警。
- **`numpilot/__main__.py`** 已改:① 新增 `doctor` 子命令;② `migrate` 前置 `_preflight`,npu 缺硬依赖 → `sys.exit(3)` 拦截;③ 加 `--skip-env-check` 逃生口。

## 🔧 进行中:Part 3(真机对比)

- **`numpilot/contracts.py`**:`VerificationReport` 已加 `backend / t_orig_ms / t_mig_ms / n_elements` 字段。
- **`numpilot/agents/verification.py`** 已重写:迁移侧对全部输入 `to_device`(H2D 移出计时、避免混合类型报错),计时循环内 `sync=backend.to_host` 强制 NPU 同步。**shim 后端已回归通过**(pass, max_rel=0)。
- **当前卡点(真机)**:迁移产物 `examples/mlp_forward.py` 里 `np.maximum(x, 0)` → `ap.maximum(x, 0)`,真机因标量 0 **FAIL**。

## ▶️ 续跑步骤(网络无关,直接干)

1. **解标量问题(二选一或都做)**:
   - (推荐)增强 AST 引擎 `numpilot/engines/ast_rewrite.py`:把 `np.maximum(x,0)`/`np.maximum(0,x)` 识别为 relu 模式 → 改写成 `ap.relu(x)`(relu 在覆盖集)。这样旧 mlp demo 真机可通。
   - 设计新的**大 GEMM demo** 时,forward 规避 asnumpy 不支持的标量算子(用 `ap.relu` 而非 `maximum(x,0)`;避免 `+标量`/`*标量`,偏置用同 shape 张量 `ap.add(dev, bias_tensor)`)。
2. **Part 3 大 GEMM demo**:新增 `examples/big_gemm.py`,纯 numpy,仅覆盖集算子,规模**非常大**(如多层 4096×4096 matmul 链 + relu),`build_inputs()` 全用 `np.linspace/zeros` 造确定性输入。
3. **orchestrator 报告渲染**:`numpilot/orchestrator.py` 的 `_render_md` / `summary` 用新字段并列 **numpy@CPU(t_orig_ms) vs asnumpy@NPU(t_mig_ms)** + 加速比 + max_rel_diff;`__main__.py` 打印同理。
4. **端到端真机验证**:`python -m numpilot migrate examples/big_gemm.py --backend npu --out out_npu/`,期望 pass、数值一致、加速比 > 1(大 GEMM 下 NPU 显著快于 numpy CPU)。

## 待办清单(TODO)
- [x] Part 1 asnumpy 真机装好
- [x] Part 2 env_check + doctor + migrate 前置拦截
- [x] Part 3a verification to_device + 同步计时(shim 已过)
- [x] Part 3b 标量迁移(np.maximum(x,0)→ap.relu)
- [x] Part 3c 大 GEMM demo(examples/big_gemm.py)
- [x] Part 3d 报告并列 CPU/NPU 时间
- [x] Part 3e 端到端真机 pass(mlp + big_gemm 均 pass)

---
---

# 📌 会话进展快照 #3(Part 3 全部完成 + asnumpy 128× 性能调查)

> 追加于 2026-07-03。本轮把 Part 3 剩余项全部做完并真机验证;另外深入排查了
> asnumpy README 的"128× 加速比"声称,结论是**不可复现,其 numpy 基线是坏的**。

## ✅ Part 3 收尾(全部完成,代码已改)

| 项 | 改动文件 | 状态 |
|----|---------|------|
| 3b relu 模式改写 | `numpilot/engines/ast_rewrite.py` | `np.maximum(x,0)`/`np.maximum(0,x)` → `ap.relu(x)`,绕开 asnumpy 标量不支持;visit_Call 拦截,递归改写内部 |
| 3c 大 GEMM demo | `examples/big_gemm.py`(新) | 多层 matmul+relu 链,_N=2048,类 He 初始化保数值稳定;⚠️ `__main__` 只用覆盖集算子(`np.absolute` 非 `np.abs`,否则分析器判缺口) |
| 3d 报告并列时间 | `numpilot/orchestrator.py` + `numpilot/__main__.py` | 报告/CLI 打印 numpy@CPU vs asnumpy@backend 并列 + 加速比 + 谁更快 |
| 3e 真机 pass | — | mlp_forward:pass, max_rel=1.49e-7;big_gemm:pass, max_rel=0.0 |

- 23 个 pytest 全过;shim/npu 双后端 smoke 全过。
- **已知非阻塞现象**:真机加速比 < 1(见下)。这不是 bug,是本机 CPU 太强 + fp32 不是 NPU 强项。

## 🔬 asnumpy 128× 性能调查(重要结论)

**背景**:asnumpy README/docs/benchmarks.md 声称 3000×3000 fp32 下 `ap.multiply()` 比
`np.multiply()` 快 **128.70×**。用户要求核实。

**做法**:原样跑官方 `examples/03_multiply.py`;并做公平计时(含 to_numpy 同步)对比;
扫描 100²→30000² 多档规模;matmul/multiply/add 都测;查了硬件。

**硬件事实(本机)**:CPU 2TB RAM + 强多核 BLAS/SIMD;NPU=Ascend910,**每卡 HBM 64GB**;
CANN 8.5.0(README 那台是 8.2.RC1)。

**核心结论**:
1. **128× 不可复现**。官方脚本本机实测:multiply 3000² 只有 0.74×,Average 0.30×、Max 0.74×,**全部 < 1**。
2. **根因在 numpy 基线**:README 说 numpy 3000² 要 **14.36 秒**;本机实测只要 **9.78ms**(差 ~1468×)。
   换算每元素:README 683~1595 ns/元素(≈4000 时钟周期/次乘法,物理不可能),本机 ~1 ns/元素
   (贴内存带宽 ~9GB/s,正常向量化)。README 那条"numpy"极可能走了非向量化/对象/病态路径。
3. **开销假设只对一半**:speedup 随规模单调爬升(1M 0.16× → 100M 0.89×),固定开销确实摊薄,
   但 **element-wise 访存受限算子在本机爬到 ~0.9× 见顶,过不了 1.0**(且这是不含同步的宽容口径;
   含 D2H 同步只有 ~0.4×)。
4. **fp16 崩溃**:`ap.matmul` fp16 触发 `double free or corruption`,NPU 强项路径当前不可用。
5. **matmul fp32**:满核 CPU 0.28×@2048、单核 0.87×@2048;到 6144 才 0.616×,始终 < 1。

**一句话**:本机环境下,fp32 任何算子 asnumpy@NPU 都打不过健康 numpy;README 的 128× 是坏掉的
分母(numpy)撑起来的。要 NPU 真赢需算力密集(大 matmul)+ 低精度(fp16/bf16,但当前会崩)+
一次搬入多次复用的算子链。

**asnumpy 仓库改动(不属 numpilot,勿推,还原用 `git checkout`)**:
- `/home/lixing/asnumpy/examples/03_multiply.py`:shapes 改 5 档扫描、迭代 3/10
- `/home/lixing/asnumpy/examples/01_add.py`:同上,测 add
- 还原:`cd /home/lixing/asnumpy && git checkout examples/01_add.py examples/03_multiply.py`

## ▶️ 下一步(demo 叙事待定,需用户拍板)

Part 3 代码已完成,但"demo 加速比 > 1"这个原始期望在本机 fp32 下达不到。候选方向:
- **A(推荐)**:如实呈现"迁移正确 + 数值一致 + 时延如实评测",报告点明 NPU 适用边界。
- **B**:继续找 NPU 能赢的负载(fp16 修复、算子融合链、更大 matmul)。
- **C**:判决性实验——本机人为造"病态 numpy"(dtype=object)复现 README 的几秒级慢,坐实其基线问题。

Phase 5/6(report.md、summary.json、session 导出)尚未做。
