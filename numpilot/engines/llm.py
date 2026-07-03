"""可插拔 LLM 迁移引擎(provider 无关, OpenAI 兼容协议)。

启用需以下环境变量(任一缺失即回退 AST 引擎):
  NUMPILOT_LLM_BASE_URL    OpenAI 兼容 endpoint, 例: https://.../v1
  NUMPILOT_LLM_AUTH_TOKEN  鉴权 token(以 Authorization: Bearer 发送)
  NUMPILOT_LLM_MODEL       模型名
  NUMPILOT_LLM_PROVIDER    (可选)仅作标签, 例: pangu | glm | openai | deepseek

只用标准库 urllib 调用, 不引入额外依赖。任何异常(网络/解析/鉴权)
都回退确定性 AST 引擎, 保证 demo 始终可跑。凭据仅从环境变量读取,
不落盘、不入库。
"""
import json
import os
import re
import urllib.request

from numpilot.contracts import OffloadPlan
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.engines.base import MigrationEngine

_REQUIRED_ENV = (
    "NUMPILOT_LLM_BASE_URL",
    "NUMPILOT_LLM_AUTH_TOKEN",
    "NUMPILOT_LLM_MODEL",
)

_SYSTEM_PROMPT = (
    "你是 NumPilot 的代码迁移智能体。把用户给的 NumPy 程序迁移为 AsNumpy(NPU)版本:\n"
    "1) 在 import numpy 之后加 `import asnumpy as ap`;\n"
    "2) 把下列需上 NPU 的算子调用 np.<op>(...) 改为 ap.<op>(...);其余保持不变;\n"
    "3) 不改变计算逻辑与函数签名, 只做命名空间替换与必要的 import。\n"
    "只输出完整的迁移后 Python 源码, 用 ```python 代码块包裹, 不要解释。"
)


class LlmEngine(MigrationEngine):
    name = "llm"

    def is_configured(self) -> bool:
        return all(os.getenv(k) for k in _REQUIRED_ENV)

    def rewrite(self, source: str, plan: OffloadPlan) -> str:
        if not self.is_configured():
            return AstRewriteEngine().rewrite(source, plan)
        try:
            npu_ops = sorted(op for op, p in plan.placement.items() if p == "NPU")
            user_msg = f"需上 NPU 的算子: {npu_ops}\n\n源码:\n```python\n{source}\n```"
            content = self._chat(_SYSTEM_PROMPT, user_msg)
            code = self._extract_code(content)
            if code and "ap." in code:
                return code
        except Exception:
            pass
        # 任何失败都回退 AST, 保证可跑
        return AstRewriteEngine().rewrite(source, plan)

    @staticmethod
    def _chat(system: str, user: str) -> str:
        base = os.environ["NUMPILOT_LLM_BASE_URL"].rstrip("/")
        url = base + "/chat/completions"
        body = json.dumps({
            "model": os.environ["NUMPILOT_LLM_MODEL"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + os.environ["NUMPILOT_LLM_AUTH_TOKEN"])
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_code(content: str) -> str:
        m = re.search(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
        return (m.group(1) if m else content).strip()
