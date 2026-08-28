"""Compatibility views over the versioned AsNumpy capability catalog."""

from numpilot.capability_catalog import ASNUMPY_CAPABILITY_CATALOG


# Historical callers consume a top-level frozenset.  Qualified/nested keys are
# intentionally available only through ``ASNUMPY_CAPABILITY_CATALOG``.
SUPPORTED_OPS = ASNUMPY_CAPABILITY_CATALOG.top_level_keys


# asnumpy 提供的"自定义 / 非 numpy 原生"算子:relu/gelu/softmax 是 numpy 没有的 nn 算子,
# 历史分类名保留兼容：这些是已有 AsNumpy 扩展 API，不代表本次迁移新生成了
# Ascend C 自定义算子；真实实现来源仍以 capability/provenance manifest 为准。
CUSTOM_OPS = frozenset({"relu", "gelu", "softmax"})


def is_supported(name: str) -> bool:
    """判断某 numpy 函数名是否被 AsNumpy 覆盖。"""
    return name in SUPPORTED_OPS


def is_custom(name: str) -> bool:
    """判断某算子是否为 AsNumpy 非 NumPy 原生扩展 API。"""
    return name in CUSTOM_OPS
