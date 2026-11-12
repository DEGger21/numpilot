"""迁移引擎策略接口。"""
from abc import ABC, abstractmethod

from numpilot.contracts import OffloadPlan


class MigrationEngine(ABC):
    name: str = "base"

    @abstractmethod
    def rewrite(self, source: str, plan: OffloadPlan, taskbook=None) -> str:
        """把 numpy 源码按卸载方案改写为 asnumpy 版本, 返回新源码。

        ``taskbook`` 是 Planner 生成的逐调用点事实; 引擎可以忽略它, 但
        configured-model 引擎应把它和 AsNumpy 代码格式一起交给模型。
        """
        raise NotImplementedError
