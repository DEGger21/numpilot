# Ascend C 算子生成、构建与 benchmark 记录（2026-11-14）

本批对 `np.spacing`、`np.diff`、`np.roll` 三个真实缺口 API 走
`resolve-gaps → complete-gaps --provider llm` 的真实生成路线，并把生成草稿继续推进到
**编译、私有 OPP 部署、NPU 原生数值对拍、AsNumpy 公共 API 接入、完整程序 CPU/NPU
benchmark**。三组最终均为 `case_local_adapter_benchmarked_not_activated`：能力只安装在
各自用例的私有目录，没有写入共享 AsNumpy 或系统 CANN，也没有激活为全局能力。

| 样例 | 缺口 API | 生成墙钟 | 生成 tokens | 编译 | NPU 原生对拍 | CPU / NPU | 加速比 | 完整端到端 |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `D1002_ulp_budget` | `np.spacing` | 641 s | 664,991 | 通过 | 逐位一致 | 34.860681 / 4.032468 ms | **8.6450x** | 2,170 s |
| `D1004_two_track_velocity` | `np.diff` | 955 s | 745,668 | 通过 | 非 NaN 位逐位一致，NaN 位置一致 | 27.339721 / 8.133022 ms | **3.3616x** | 3,102 s |
| `D1009_periodic_transport` | `np.roll` | 890 s | 716,941 | 通过 | 逐位一致 | 26.669054 / 7.455630 ms | **3.5770x** | 3,112 s |

三组合计 **2,127,600 tokens**；`D1004`/`D1009` 两路并行生成，`D1002` 随后补跑。
完整端到端从该样例开始生成算子算起，包含生成、编译/修复、部署、NPU 对拍、适配层构建、
完整程序迁移和 CPU/NPU benchmark，不包含人工阅读代码的空闲时间。

## 生成后确定性验证

- **编译**：三份草稿均成功生成 `custom_opp_ubuntu_aarch64.run` 和 Ascend C kernel binary。
- **部署**：安装到 `<case>/numpilot-work/asc_operator/custom-v2/deploy-vN/install/vendors/<op>_custom`。
- **NPU 原生对拍**：使用 `tools/aclnn_generated_probe.py` 直接加载 `libcust_opapi.so`，
  创建 ACL tensor 并调用两段式 ACLNN API；`spacing`/`roll` 逐位一致，`diff` 非 NaN 位逐位一致。
- **适配层**：`asnumpy.spacing` / `asnumpy.diff` / `asnumpy.roll` 均从 `asnumpy.custom_ops`
  路由到生成的 ACLNN 入口，不是 `numpy` alias；wheel 只安装到用例私有 `stage/`。
- **完整程序**：`benchmark_case_perf.py` 对 1024×1024 输入跑 3 次 warmup + 20 次采样，
  三组 `comparison.consistent=true`，CPU/NPU 中位数与加速比如上表。

## 通用框架修复（不是样例定向）

1. `full_like`、`ones_like`、`zeros_like`、`empty_like` 的 `dtype` 改为可选；省略时从原型
   数组继承。修复位于 `numpilot/agents/analysis.py` 的统一调用契约，覆盖所有 dtype-less
   `*_like` 程序；回归测试 131 passed。
2. 生成提示词增加跨文件不变量：TilingData 与 kernel entry 各一份权威定义；host 只能写
   header 声明的字段；`TilingContext::GetInputShape/GetOutputShape` 返回
   `const gert::StorageShape*`，必须调用 `GetStorageShape()`；kernel 模板参数必须与
   `ASCENDC_TPL_ARGS_DECL` 一致；使用 `l0op::` 时必须把 CANN `opapi` 链入 `cust_opapi`。
3. 新增 `tools/audit_generated_asc_contract.py`，只读检查 TilingData 字段一致性、kernel
   模板参数、shape 指针类型和 opapi 链接；可在生成后、编译前做确定性门禁。
4. AsNumpy 适配层构建发现 `full_like(..., dtype=None)` 运行期仍把 `None` 传给 pybind；
   在适配层源码树中统一改为从原型 dtype 继承，回归到 AsNumpy 文档声明的语义。
5. `roll` 是纯数据搬运算子，生成 kernel 的 `Muls(x, 1.0f)` 会把 NaN payload 规范化；
   改为 UB→UB `DataCopy`，保证 NaN / ±Inf / ±0 的位级不变。

## 产物结构

每个 `<case>/asc_operator/` 包含：

- `numpy_source.py`：放大的 NumPy 源程序（1024×1024 float32）；
- `GAP_TASKBOOK.md/json`、`gap-spec.json`：缺口任务书与契约；
- `draft/`：模型生成并经确定性编译修复的 Ascend C 源码、host/kernel/op_api、构建脚本和单测；
- `repairs/`：修复前后哈希清单与原始草稿；
- `build/`、`deploy/`：编译和私有 OPP 安装回执；
- `verify/`：NPU 原生对拍与跨文件契约审计；
- `adapter/`：AsNumpy 接入源码、wheel 和构建日志；
- `benchmark/`：完整程序 `migrated.py`、公共路由校验、CPU/NPU 结果；
- `metrics.json`：机器可读指标；
- `README.md`：本样例摘要。

## 口径边界

- 模型生成的是 Ascend C 草稿；构建、NPU 对拍、适配层和 benchmark 是确定性验证，不是模型自报。
- 三组均未调用模型做最终“成功”结论，也没有把人工修复描述成模型生成。
- 适配层与 OPP 仅存在于用例私有目录；没有写入共享 AsNumpy / 系统 CANN，也没有全局激活。
- `diff` 的 NaN payload 与 CPU 不同，但 NaN 位置和非 NaN 位一致，按语义等价记录。
