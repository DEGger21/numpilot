# 全覆盖接口镜像（71 例）

本目录是 `../cases/<group>/<case>/program.py` 的只读镜像副本视图，不修改原始 benchmark
语料。分类规则：`forward` 及其可达 helper 中的 `np.*` 接口全部出现在固定 AsNumpy 导出表，
或属于文档化别名（例如 `np.abs → asnumpy.absolute`、`np.acos → arccos`）。

`J_out_of_subset` 中有 8 例 API 表面覆盖，但它们仍是静态拒绝用例；逐例的
`out_of_scope` 标记见 `../classification.json`。
