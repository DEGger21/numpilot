# NumPilot 比赛证据索引

本目录只保留当前比赛主叙事直接引用的两份 Ascend 真机证据。历史 profiler 数据库、原始
日志、数值数组、失败轮次和已否决的 MatMul 展示材料已从发布分支移除；它们仍可从开发分支
与 Git 历史恢复，但不会进入提交包。

> 当前两份回执均记录 `git_dirty=true`，所以它们是可审计的**开发态证据**，不是最终 RC
> 证据。最终提交前必须在冻结提交上重跑主案例，用干净回执替换，并重新生成本目录清单。

## 当前证据

| 案例 | 工作负载 | Resident 中位加速 | 保守加速 | 发布边界 |
|---|---|---:|---:|---|
| [Poisson-Ritz 应用阶段](competition-poisson-stage-20260828/README.md) | 9,000,000 个 `float32` 配点；解析源项与 SmoothL1 残差图 | **2.567×** | 2.471× | 非完整求解器；一次组件和 0.789×，不是直接 E2E |
| [逐元素优势区间](competition-ordinary-20260828/README.md) | 9,000,000 个 `float32` 元素；设备驻留重复逐元素链 | **8.079×** | 7.503× | 合成压力链；一次组件和 1.220×，不是直接 E2E |

两条链路都经过：

1. 当前 AsNumpy/CANN/SoC 组合上的 `DIRECT` 能力分析；
2. configured `glm-5.3` 实际远程迁移；
3. exact AST 结构门和独立进程数值对拍；
4. CPU/strict-NPU 分离 worker、相同 CPU 亲和核、CPU BLAS 单线程；
5. runtime identity、调用次数、逐样本同步与最终 D2H 门；
6. 原始样本、统计量和文件哈希写入结构化回执。

## 路径与哈希脱敏边界

公开回执不暴露开发机用户名、工作区、虚拟环境、CANN 安装目录或临时目录。源文件路径保留为
仓库相对路径；运行时路径统一显示为 `/REDACTED/...`。脱敏只改变展示元数据，不改变源码、迁移
产物、计时样本、输出摘要或组件文件哈希；受影响的 runtime identity、migration report、bootstrap
引用及 case/root `MANIFEST.sha256` 已按最终公开字节重新计算。因而回执仍可由独立审计器验证，
但不能仅凭公开包重建开发机目录。

## 如何复核

根清单覆盖本索引和两个案例目录中的全部文件：

```bash
sha256sum -c evidence/MANIFEST.sha256
```

独立审计 benchmark 回执：

```bash
python tools/audit_script_benchmark_receipt.py \
  evidence/competition-poisson-stage-20260828

python tools/audit_script_benchmark_receipt.py \
  evidence/competition-ordinary-20260828
```

测试同时检查清单完整性、目录白名单、案例内自清单以及禁带中间产物：

```bash
python -m pytest -q \
  tests/test_public_evidence.py \
  tests/test_competition_ordinary_evidence.py
```

## 统计口径

- `resident speedup` 只统计输入已驻留设备后的重复计算；
- 冷启动、一次输入准备和最终 D2H 不混入 headline；
- `component_sum` 是诊断性组件相加，回执明确标记
  `directly_sampled_end_to_end=false`；
- `conservative speedup` 使用 CPU p10 / NPU p90；
- 所有结论只适用于回执绑定的 dtype、shape、硬件、CANN、AsNumpy 与代码身份。

NumPilot 不把未采样的口径写成实测 E2E，也不把合成优势区间外推为通用应用性能。
