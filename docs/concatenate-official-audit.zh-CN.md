# np.concatenate 官方 ACLNN 候选审计（2026-11-03）

本文记录 `numpy.concatenate` 缺口在目标机器上的 official-first 解析结果。解析过程只读检查已安装
CANN 的头文件和动态库，不读取 `llm.json`、不调用模型、不联网、不编译源码、不修改 AsNumpy 组件。

## 固定目标

- Qualified API：`numpy.concatenate`
- 目标 SoC：Ascend910B3
- CANN 版本：9.2.0-beta.2
- 已安装 CANN 根：由 `--cann-root` 显式指定；本文不记录私有绝对路径
- 固定 AsNumpy 组件：`0.4.1.dev12+gc9115e8ce.d20261031`，wheel SHA-256
  `a8c47cfe71994f3bfd4180d7354df3904d2bc96c476c3119083fefac747a2138`

## 只读证据

| 工件 | 证据 |
|---|---|
| `include/aclnnop/aclnn_cat.h` | SHA-256 `79686cc566a1b242cd7a656aa633aa6e255b0bd636f27f3ddd7cce0adfca56fd`，1355 bytes |
| `lib64/libopapi.so` | SHA-256 `0b93d5b6cdfda98f1176d1703fafdbc513d87d5b805ac7984d617a209615e58b`，53791584 bytes，2609 个 `aclnn*` 动态符号 |
| `lib64/libopapi_math.so` | SHA-256 `19bfa9a2b86bfe48e7113c6a98e4220ad12bbf42da8786b3956321515c036db7`，同时定义两个所需符号 |
| `lib64/libopapi_nn.so` | 不定义 `aclnnCat` / `aclnnCatGetWorkspaceSize`；因此不在该映射的库优先级内 |

头文件声明两段式接口：

```c
aclnnStatus aclnnCatGetWorkspaceSize(
    const aclTensorList* tensors, int64_t dim, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnCat(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, const aclrtStream stream);
```

`nm -D --defined-only` 在 `libopapi.so` 和 `libopapi_math.so` 中都确认
`aclnnCatGetWorkspaceSize` 与 `aclnnCat` 为已定义动态符号。`aclTensorList` 由
`include/aclnn/acl_meta.h` 的 `aclCreateTensorList` / `aclDestroyTensorList` 管理。

## 解析回执

`resolve-gaps` 生成 write-once 回执，绑定源程序、GapSpec、目标 SoC/CANN 和上述工件摘要：

- `resolution_id`：`be2e2e7b5da4365d80d1f67c9c9f6c7691806ca7fc76ef3968fd7d3af1a5d623`
- 回执文件 SHA-256：`7cafe59a602f1c1607fb668e183561b7dab867bab39983c9b7face5be3406bbf`
- `status`：`official_candidate_found`
- `route`：`installed_cann_adapter_candidate`
- `installed_cann.status`：`exact_candidate`
- `adapter_generation_required`：`true`
- `semantic_audit_required`：`true`
- `required_next_gate`：`generate_and_qualify_asnumpy_adapter`

该回执只证明已安装 CANN 中存在与审计映射一致的 ABI 和动态符号，不证明语义等价，也不授权
构建、加载、集成、激活或性能结论。

## 审计映射的边界

Catalog revision `installed-cann-aclnn-adapter-candidates-v3` 新增
`numpy.concatenate-aclnn-cat`，只绑定 CANN `9.2.0-beta.2` 和 Ascend910B3，不复用 MatMul 的
`9.0.0` / Ascend910B4 审计画像。审查意见限定：

1. 只映射显式整数 `axis` 到 ACLNN Cat 的 `dim`；`axis=None` 以及 `dtype`、`out`、`casting`
   关键字不进入原生路径。
2. 只接收非空、同 rank、非拼接维 extents 相同、C-contiguous 的 float32 设备张量序列，并按
   预计算输出 shape 显式分配结果。
3. 在集成前必须用 NumPy oracle 验证负 axis 归一化、单输入拷贝、空序列、混合 dtype、shape
   不匹配和非连续输入。

## 复现方式

```bash
export CANN_ROOT=/path/to/installed/cann
python -m numpilot resolve-gaps \
  benchmark/cases/I_gap_model/I006_concatenate_axis/program.py \
  --target-soc Ascend910B3 \
  --cann-version 9.2.0-beta.2 \
  --cann-root "$CANN_ROOT" \
  --out /absolute/private/concatenate-resolution.json
```

复现时必须比较完整 `resolution_id`，不能只比较 `status`。任何头文件、库、CANN 版本或 SoC 漂移
都会使该回执失效，需要重新审计。
