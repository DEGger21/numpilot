"""可插拔 LLM 迁移引擎(provider 无关, OpenAI 兼容协议)。

启用需以下环境变量:
  NUMPILOT_LLM_BASE_URL    OpenAI 兼容 endpoint, 例: https://host:port/v1(务必带 /v1)
  NUMPILOT_LLM_AUTH_TOKEN  鉴权 token(以 Authorization: Bearer 发送)
  NUMPILOT_LLM_MODEL       模型名, 例: glm-5.1
  NUMPILOT_LLM_PROVIDER    (可选)仅作标签, 例: pangu | glm | openai | deepseek
  NUMPILOT_LLM_MAX_TOKENS  (可选)生成上限, 默认 8192(部分网关强制要求此字段)
  NUMPILOT_LLM_INSECURE    (可选)"1"=显式跳过 TLS 证书校验(仅限受信内网), 默认 "0"
  NUMPILOT_LLM_TIMEOUT     (可选)单次请求超时秒, 默认 120

两种模式(见 __init__ strict):
  - strict=False(默认, 供编程/测试): 未配置或调用失败 → 回退确定性 AST, 保证可跑。
  - strict=True(CLI 显式 --engine llm): 未配置/无响应/返回非法 → 抛 LlmError 直接终止,
    不回退 —— 保证"用智能体迁移"名副其实, 失败即暴露, 不被 AST 掩盖。
凭据仅从环境变量读取, 不落盘、不入库。
"""
import json
import os
import re
import ssl
import urllib.request

from numpilot.contracts import OffloadPlan
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.engines.base import MigrationEngine


class LlmError(RuntimeError):
    """strict 模式下 LLM 未配置/无响应/返回非法迁移码时抛出, 由 CLI 捕获并终止。"""


_REQUIRED_ENV = (
    "NUMPILOT_LLM_BASE_URL",
    "NUMPILOT_LLM_AUTH_TOKEN",
    "NUMPILOT_LLM_MODEL",
)

_SYSTEM_PROMPT = (
    "你是 NumPilot 的代码迁移智能体。把用户给的 NumPy 程序迁移为 AsNumpy(NPU)版本。\n"
    "AsNumpy 用 `import asnumpy as ap`, API 与 NumPy 基本一致, 但有以下**硬约束**必须遵守:\n"
    "1) 二元算子(add/subtract/multiply/divide/matmul/maximum/minimum/power 等)"
    "**两侧都必须是 NPUArray, 不接受 Python 标量**。\n"
    "2) 因此 `np.maximum(x, 0)` 或 `np.maximum(0, x)`(即 ReLU)**必须改写为 `ap.relu(x)`**, "
    "不能写成 `ap.maximum(x, 0)`(会因标量 0 报错)。\n"
    "3) 其余 `np.<op>(...)` 直接改为 `ap.<op>(...)`;`np` 命名空间的 import 保留。\n"
    "4) 只迁移 `forward(...)` 计算主体;`build_inputs()` 保持用 numpy 构造(它产出 CPU 输入)。\n"
    "5) 在 forward 入口为每个数组参数加入幂等边界:`if isinstance(x, np.ndarray): "
    "x = ap.ndarray.from_numpy(x)`。设备数组和标量必须保持原样。\n"
    "6) 不改变计算逻辑、函数名与签名, 保证与原程序数值等价。\n"
    "只输出完整的迁移后 Python 源码, 用 ```python 代码块包裹, 不要任何解释文字。"
)


class LlmEngine(MigrationEngine):
    name = "llm"

    def __init__(self, strict: bool = False):
        self.strict = strict

    def is_configured(self) -> bool:
        return all(os.getenv(k) for k in _REQUIRED_ENV)

    def rewrite(self, source: str, plan: OffloadPlan) -> str:
        if not self.is_configured():
            if self.strict:
                missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
                raise LlmError("LLM 引擎未配置, 缺环境变量: " + ", ".join(missing)
                               + "。请 `set -a; . ./.env.llm; set +a` 后重试。")
            return AstRewriteEngine().rewrite(source, plan)

        model = os.environ["NUMPILOT_LLM_MODEL"]
        npu_ops = sorted(op for op, p in plan.placement.items() if p == "NPU")
        user_msg = f"需上 NPU 的算子: {npu_ops}\n\n源码:\n```python\n{source}\n```"
        try:
            content = self._chat(_SYSTEM_PROMPT, user_msg)
            code = self._extract_code(content)
        except Exception as e:
            if self.strict:
                raise LlmError(f"调用模型 {model} 失败(无响应/网络/解析错误): "
                               f"{type(e).__name__}: {str(e)[:200]}") from e
            return AstRewriteEngine().rewrite(source, plan)

        if code and "ap." in code:
            return code
        if self.strict:
            raise LlmError(f"模型 {model} 返回未包含有效迁移代码(缺少 ap.* 调用), "
                           "无法确认迁移成功。")
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
            "max_tokens": int(os.getenv("NUMPILOT_LLM_MAX_TOKENS", "8192")),
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + os.environ["NUMPILOT_LLM_AUTH_TOKEN"])
        # 默认校验证书；只有受信内网自签网关才可显式选择 insecure。
        ctx = (ssl._create_unverified_context()
               if os.getenv("NUMPILOT_LLM_INSECURE", "0") == "1" else None)
        timeout = float(os.getenv("NUMPILOT_LLM_TIMEOUT", "120"))
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_code(content: str) -> str:
        m = re.search(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
        return (m.group(1) if m else content).strip()
