"""可插拔 LLM 迁移引擎(provider 无关, OpenAI 兼容协议)。

CLI 会优先读取以下环境变量，并可通过 ``numpilot configure-llm`` 自动加载
框架管理的私有用户配置:
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
真实凭据不入库；框架管理的用户配置以 0600 权限保存在用户配置目录。
"""
import hashlib
import json
import os
import re
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from numpilot.contracts import OffloadPlan
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.engines.base import MigrationEngine


class LlmError(RuntimeError):
    """strict 模式下 LLM 未配置/无响应/返回非法迁移码时抛出, 由 CLI 捕获并终止。"""


class LlmHttpError(RuntimeError):
    """A redacted HTTP failure that never includes the URL or response body."""

    def __init__(self, status_code: int):
        if (
            isinstance(status_code, bool)
            or not isinstance(status_code, int)
            or status_code < 100
            or status_code > 599
        ):
            status_code = 0
        self.status_code = status_code
        super().__init__("remote LLM HTTP status %d" % status_code)


class LlmTransportError(RuntimeError):
    """A redacted transport failure that does not expose endpoint details."""

    def __init__(self, category: str):
        safe_category = re.sub(r"[^A-Za-z0-9_.-]", "?", category)[:128]
        super().__init__(
            "remote LLM transport failed: %s" % (safe_category or "unknown")
        )


class LlmResponseError(RuntimeError):
    """A safe parse failure carrying the receipt for a successful HTTP response."""

    def __init__(self, reason: str, receipt: Dict[str, object]):
        self.receipt = dict(receipt)
        super().__init__("remote LLM response is invalid: %s" % reason)


_REQUIRED_ENV = (
    "NUMPILOT_LLM_BASE_URL",
    "NUMPILOT_LLM_AUTH_TOKEN",
    "NUMPILOT_LLM_MODEL",
)

_REQUEST_ID_HEADERS = (
    "x-request-id",
    "request-id",
    "x-openai-request-id",
    "x-zhipu-request-id",
    "x-trace-id",
)
_MAX_RESPONSE_BODY_BYTES = 3 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_receipt_text(
    value: object,
    *,
    token: str,
    base_url: str,
    limit: int = 512,
) -> Optional[str]:
    """Bound untrusted response metadata and redact known client secrets."""

    if value is None:
        return None
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    text = str(value).strip()
    for sensitive in sorted({token, base_url}, key=len, reverse=True):
        if sensitive:
            text = text.replace(sensitive, "redacted")
    text = "".join(
        character
        if (
            32 <= ord(character) < 127
            and (character.isalnum() or character in "._:-")
        )
        else "?"
        for character in text
    )
    return text[:limit] or None


def _request_id(headers: object, *, token: str, base_url: str) -> Optional[str]:
    try:
        normalized = {
            str(name).lower(): value
            for name, value in headers.items()
        }
    except (AttributeError, TypeError, ValueError):
        return None
    for name in _REQUEST_ID_HEADERS:
        if name in normalized:
            return _safe_receipt_text(
                normalized[name], token=token, base_url=base_url
            )
    return None


def _integer_usage_items(value: object, *, token: str) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for raw_name, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
        name = str(raw_name)
        if (
            len(result) >= 32
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", name)
            or (token and token in name)
            or isinstance(raw_value, bool)
            or not isinstance(raw_value, int)
            or raw_value < 0
            or raw_value > 2 ** 63 - 1
        ):
            continue
        result[name] = raw_value
    return result

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
                               + "。请运行 `python -m numpilot configure-llm` 后重试。")
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
        """Compatibility wrapper returning only model content."""

        content, _receipt = LlmEngine._chat_with_receipt(system, user)
        return content

    @staticmethod
    def _chat_with_receipt(
        system: str, user: str
    ) -> Tuple[str, Dict[str, object]]:
        """Perform one real request and return content plus a redacted receipt."""

        base = os.environ["NUMPILOT_LLM_BASE_URL"].rstrip("/")
        token = os.environ["NUMPILOT_LLM_AUTH_TOKEN"]
        requested_model = os.environ["NUMPILOT_LLM_MODEL"]
        url = base + "/chat/completions"
        request_data = {
            "model": requested_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": int(os.getenv("NUMPILOT_LLM_MAX_TOKENS", "8192")),
        }
        body = json.dumps(request_data).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + token)
        # 默认校验证书；只有受信内网自签网关才可显式选择 insecure。
        insecure = os.getenv("NUMPILOT_LLM_INSECURE", "0") == "1"
        tls_verification_enabled = base.lower().startswith("https://") and not insecure
        ctx = ssl._create_unverified_context() if insecure else None
        timeout = float(os.getenv("NUMPILOT_LLM_TIMEOUT", "120"))
        started_at = _utc_now()
        try:
            response = urllib.request.urlopen(req, timeout=timeout, context=ctx)
        except urllib.error.HTTPError as exc:
            status_code = (
                exc.code
                if (
                    not isinstance(exc.code, bool)
                    and isinstance(exc.code, int)
                )
                else 0
            )
            try:
                exc.close()
            except Exception:
                pass
            raise LlmHttpError(status_code) from None
        except Exception as exc:
            raise LlmTransportError(type(exc).__name__) from None
        try:
            with response as resp:
                status = getattr(resp, "status", None)
                if status is None:
                    status = resp.getcode()
                if isinstance(status, bool) or not isinstance(status, int):
                    raise LlmTransportError("invalid_http_status")
                if status < 200 or status >= 300:
                    raise LlmHttpError(status)
                headers = getattr(resp, "headers", {})
                response_body = resp.read(_MAX_RESPONSE_BODY_BYTES + 1)
                if len(response_body) > _MAX_RESPONSE_BODY_BYTES:
                    raise LlmTransportError("response_too_large")
        except (LlmHttpError, LlmTransportError):
            raise
        except Exception as exc:
            raise LlmTransportError(type(exc).__name__) from None
        ended_at = _utc_now()
        receipt = {
            "schema_version": 1,
            "utc_start": started_at,
            "utc_end": ended_at,
            "http_status": status,
            "requested_model": _safe_receipt_text(
                requested_model, token=token, base_url=base
            ),
            "response_model": None,
            "response_id": None,
            "response_created": None,
            "request_body_sha256": hashlib.sha256(body).hexdigest(),
            "response_body_sha256": hashlib.sha256(response_body).hexdigest(),
            "content_sha256": None,
            "request_id": _request_id(headers, token=token, base_url=base),
            "usage": {},
            "tls_verification": (
                "enabled" if tls_verification_enabled else "disabled"
            ),
        }
        try:
            data = json.loads(response_body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise LlmResponseError("invalid_json", receipt) from None
        if not isinstance(data, dict):
            raise LlmResponseError("non_object_json", receipt)
        receipt.update({
            "response_model": _safe_receipt_text(
                data.get("model"), token=token, base_url=base
            ),
            "response_id": _safe_receipt_text(
                data.get("id"), token=token, base_url=base
            ),
            "response_created": (
                data.get("created")
                if (
                    not isinstance(data.get("created"), bool)
                    and isinstance(data.get("created"), int)
                    and data.get("created") >= 0
                )
                else None
            ),
            "usage": _integer_usage_items(data.get("usage"), token=token),
        })
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LlmResponseError("missing_content", receipt) from None
        if not isinstance(content, str):
            raise LlmResponseError("non_text_content", receipt)
        receipt["content_sha256"] = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()
        return content, receipt

    @staticmethod
    def _extract_code(content: str) -> str:
        m = re.search(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
        return (m.group(1) if m else content).strip()
