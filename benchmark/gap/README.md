# 有缺口镜像（79 例）

本目录是 `../cases/<group>/<case>/program.py` 的只读镜像副本视图，包含三类：

- D1/E/F/G 的部分缺口与签名契约差异；
- H/I/K 的官方组合或模型缺口；
- B 组的 `out/where/casting` 契约差异，以及 J 组中未通过 API 表面检查的拒绝用例。

逐例的 `missing_apis`、`gap_reasons`、`out_of_scope` 和原始路径见
`../classification.json`。迁移产物应写回原始 `cases/<group>/<case>/` 下，而不是本视图。
