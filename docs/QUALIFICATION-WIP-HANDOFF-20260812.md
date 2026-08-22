# NumPilot Qualification / Registry Overlay 留档（2026-08-12 UTC，续接更新）

## 当前结论

本次从 `docs/GAP-COMPLETION-HANDOFF-20260812.md` 的断点继续，已实现并测试下一阶段的
“离线资格证据准入 + 原子 capability overlay”。资格准入代码本身已经完成本轮安全收尾，
但仍**尚未接入 profile-aware runtime，也没有任何算子拿到真实资格证据，尚未提交 Git**。

`numpy.cbrt` 草稿仍是 `generated_unverified`，没有被编译、执行、qualified 或 activated；
不要把“准入校验器测试通过”解读为真实 NPU 闭环已经通过。

## 本次已确认的基线

续接后首先执行：

```text
python3 -m pytest -q
144 passed in 4.27s
```

新增 qualification 文件后、继续加固前执行过相关旧回归：

```text
python3 -m pytest -q tests/test_gap_completion.py tests/test_capabilities.py
29 passed in 0.22s
python3 -m compileall -q numpilot
git diff --check
```

续接完成严格正反测试后执行：

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  tests/test_gap_qualification.py tests/test_capabilities.py
81 passed in 1.24s
```

最终全量回归结果见本文件末尾；定向通过只证明准入逻辑按测试契约工作，不证明外部
报告是真实的，也不证明任何 NPU 算子已经可用。

## 本轮实现

新增：`numpilot/gaps/qualification.py`

当前文件已包含：

- 严格、带重复 key 拒绝和大小上限的 JSON 解析；
- evidence root、中间目录、相对路径、symlink、普通文件和权限检查；
- `GateEvidence`、`RuntimeBinding`、`QualificationEvidence`；
- 六个 hard gate：artifact integrity、build、UT、NumPy 数值、AsNumpy public API、
  NPU execution；
- GapSpec、草稿 manifest、源文件、报告、运行时产物和 OPP manifest 的 SHA-256 绑定；
- manifest workspace、内容派生 artifact ID、gate claims 和 constraints hash 的精确绑定；
- CANN / SoC / Python / AsNumpy version、audit ID 和 module hash 漂移拒绝；
- `CUSTOM_NPU` capability 的派生；
- `QualificationRef`、`CapabilityOverlay`、全量验证后一次性合并；
- 带文件锁、revision、parent hash、CAS、临时文件、`fsync`、`os.replace` 的原子发布；
- 发布后目录同步/关闭/解锁失败时返回“新版本可能已可见”及 published SHA，而不是
  错报成确定未发布；
- 对 dataclass 直接构造、`bool` 冒充整数、额外 JSON 字段、宽权限目录和 runtime module
  未绑定等绕过均失败关闭；
- GapSpec 顶层/callsite/signature 精确 schema 与 `source_path` 身份绑定，拒绝 `NaN`、
  `Infinity`、隐式字符串化和重复 constraints；
- 证据读取基于已打开 fd 做 `nofollow`、文件类型、权限、大小和累计总量校验，避免 FIFO
  阻塞、无界读取并缩小 TOCTOU 窗口；
- overlay 既有目标与锁文件同样校验类型、权限和单硬链接，双重发布后异常保留首要
  durability 错误及已发布 SHA。

安全边界保持不变：该模块不调用 subprocess，不执行 `build.sh` / CMake / 草稿测试，
也不 import 生成的扩展。build 与真机报告必须来自仓库外的固定可信 harness；NumPilot
只做准入校验。哈希只能证明绑定与防漂移，不能单独证明报告真实，evidence root 必须由
可信 CI/操作者控制，后续应增加签名 attestation。

## 已覆盖的失败路径

`tests/test_gap_qualification.py` 覆盖完整正向准入，以及缺 gate、gate 非 passed、三类 CPU
fallback、spec/manifest/source/artifact/report/runtime `.so` 篡改、环境漂移或 unknown、
路径穿越、symlink、重复 key、额外字段、宽权限文件/中间目录、`bool → int`、禁止执行
生成脚本、overlay all-or-nothing、CAS/revision/parent hash、非法直接构造和发布前/发布后
I/O 异常。

## 仍未完成的真实闭环

1. 当前尚未接入 `NpuBackend`、`AnalysisAgent`、CLI 或静态 `coverage.py`。正确顺序是：
   runtime public API callable 且完整 evidence 重新验证通过后，profile-aware overlay 才能把
   unknown provenance 提升为 `CUSTOM_NPU`；不能先把 `cbrt` 手工加进静态覆盖集。
2. constraints 当前是可审计、被哈希绑定的限定文本；name-only capability 查询无法判断
   dtype/shape，因此保持 fail closed。接线前需实现 profile-aware matcher，不能把只认证
   `float32` 的算子泛化为全部 dtype/shape。
3. 尚无可信 harness 生成的 build、UT、NumPy oracle、AsNumpy public API 和 NPU execution
   六门证据，也没有签名 attestation。现有正向测试 fixture 不是生产证据。
4. `numpilot/gaps/qualification.py.orig` 是未跟踪备份；本轮没有擅自删除，提交前由操作者
   确认是否保留。
5. 未经用户明确授权不要提交 Git。

## cbrt 草稿审计结论

持久草稿：

```text
/home/ma-user/work/numpilot-gap-runs/cbrt-20260812/
```

当前草稿不可作为真实 build/UT/NPU 证据：

- `build.sh` / CMake 是 no-op 占位；
- adapter 总是可落到 CPU `np.cbrt`；
- ACLNN executor/launch 与 pybind runtime/返回路径未实现；
- kernel 有 TilingData 可见性、未赋值 `elementsPerBlock_`、tile 偏移等错误；
- 直接 `pow(x, 1/3)` 对负数不符合 `numpy.cbrt`。

本机 CANN 9.0.0 头文件已确认：

- 正确 tensor API 是 `AscendC::Power`，不是草稿里的 `Pow`：
  `/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/include/adv_api/math/power.h`；
- 910B tensor + scalar + count 签名在该文件约 392-396 行；
- `Abs` 位于 `basic_api/kernel_operator_vec_unary_intf.h`；
- `Sign` 位于 `adv_api/math/sign.h`；
- `kernel_operator.h` 不自动包含 Power/Sign，高阶聚合头是 `adv_api/kernel_api.h`；
- 可行普通值公式是 `Sign(x) * Power(Abs(x), 1/3)`，但 `-0` signbit、±Inf、NaN
  仍必须通过专门处理和真实 NPU NumPy oracle 验证；建议先只承诺 float32。

不要运行当前草稿的 `build.sh`，也不要使用 `numpyexample/build_custom_ops.sh` 原样安装；
后者包含联网、删除重建和环境修改，只能作为接线参考。

## 当前机器阻断

```text
npu-smi: symbol lookup error:
/usr/local/Ascend/driver/lib64/driver/libdrvdsmi_host.so:
undefined symbol: drvSetDeviceInfo

asnumpy_spec=None
```

因此本机目前不能诚实完成 NPU ST/public AsNumpy API/运行时 trace。不要伪造通过证据。

## 最终本地验证

```text
qualification + capabilities 定向：99 passed
全量：253 passed in 6.70s
python3 -m compileall -q numpilot：通过
bash -n run_demo.sh：通过
python3 -m numpilot --help：通过
git diff --check：通过
```

全量测试使用 `PYTHONDONTWRITEBYTECODE=1` 和 `-p no:cacheprovider`，没有联网。另对实际
LLM Token/Base URL 做值级扫描：仓库命中为 0；最终联网回执也不包含二者。

## 下次第一步

```bash
cd /home/ma-user/work/numpilot
git status --short
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

下一阶段应在可信、隔离的真实 NPU harness 产出证据，再做 profile-aware runtime 接线。
工作树包含用户原有 LLM 配置、wizard、README、run_demo 等未提交改动，全部保留；禁止
破坏性 reset。
