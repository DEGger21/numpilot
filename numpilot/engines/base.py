"""迁移引擎策略接口。"""
from abc import ABC, abstractmethod

from numpilot.contracts import OffloadPlan


class MigrationEngine(ABC):
    name: str = "base"

    @abstractmethod
    def rewrite(self, source: str, plan: OffloadPlan) -> str:
        """把 numpy 源码按卸载方案改写为 asnumpy 版本, 返回新源码。"""
        raise NotImplementedError
