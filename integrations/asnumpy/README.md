# AsNumpy 集成参考

NumPilot 本身不捆绑或自动修改 AsNumpy。本目录只保留一份经过代码审查的参考补丁，用于说明
如何减少 native 算子结果在 public wrapper 层的冗余设备内拷贝；补丁不会由 `install`、
`bootstrap`、`analyze` 或 `migrate` 隐式应用。

## `0001-move-private-native-results.patch`

补丁基于 AsNumpy commit `21368abf9`，包含 consuming constructor、Python 包装分流和对应
回归测试。它只影响内部 `_core.ndarray` 临时结果的包装过程：

- `asnumpy.ndarray(existing_public_array)` 继续使用拷贝语义；
- 仅内部包装层可以用 keyword-only `_move=True` 消费 native 临时对象；
- `_move=False` 显式拒绝，普通单参数构造不会意外进入 consuming overload；
- 算子数学语义、dtype 和 public API 不变。

在与该补丁基线一致的 AsNumpy checkout 中复核并应用：

```bash
git apply --unidiff-zero --check \
  /path/to/numpilot/integrations/asnumpy/0001-move-private-native-results.patch
git apply --unidiff-zero \
  /path/to/numpilot/integrations/asnumpy/0001-move-private-native-results.patch
```

重新构建后，必须执行补丁自带回归、所用算子的真机数值测试以及 NumPilot runtime identity
审计。NumPilot 只接受 capability manifest 中版本、来源和运行文件哈希完全匹配的 AsNumpy；
自行构建但身份不同的 wheel 会失败关闭。

这份补丁是集成参考，不是性能提升的独立证明。性能结论必须由目标运行产生的 benchmark 回执
绑定精确源码、组件、硬件和计时口径。
