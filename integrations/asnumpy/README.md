# AsNumpy 集成补丁

此目录保存 NumPilot Poisson P0 在 AsNumpy 包装层消除逐返回值设备深拷贝所需的最小补丁。
补丁基于 AsNumpy commit `21368abf9` 制作，只包含 consuming constructor、Python 分流和
对应回归测试，不包含本机 `/home/ma-user/work/asnumpyoplogin` 中其他已有修改。

## 语义边界

- `asnumpy.ndarray(existing_public_array)` 仍调用拷贝构造，不消费调用方对象；
- 只有 Python 内部包装 private `_core.ndarray` 临时结果时传入 keyword-only
  `_move=True`；
- `_move=False` 被显式拒绝，普通单参数构造不能意外选择 consuming overload；
- 该修复只消除 native result → public wrapper 的冗余 D2D 深拷贝，不改变算子计算。

## 应用

在 AsNumpy 仓库根执行：

```bash
git apply --unidiff-zero --check /path/to/numpilot/integrations/asnumpy/0001-move-private-native-results.patch
git apply --unidiff-zero /path/to/numpilot/integrations/asnumpy/0001-move-private-native-results.patch
```

重新构建 wheel 后，应至少执行补丁新增的测试，以及 creation、ndarray protocol 和所用
算子的真机测试。NumPilot 只接受 manifest 中版本与五个运行文件 SHA-256 全部匹配的 wheel；
自行构建但身份不同的 wheel 会失败关闭。

修复前后 profiler、方向探针、资格化与 250 轮回归见
[`../../evidence/poisson-ritz-p0-20260821/profiler-copy-elision/`](../../evidence/poisson-ritz-p0-20260821/profiler-copy-elision/)
和
[`../../docs/evidence/poisson-ritz-profiler-p0-20260822.md`](../../docs/evidence/poisson-ritz-profiler-p0-20260822.md)。
